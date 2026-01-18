import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
class DirState:
    """Record directory and metadata state for fast access.

    A dirstate is a specialised data structure for managing local working
    tree state information. Its not yet well defined whether it is platform
    specific, and if it is how we detect/parameterize that.

    Dirstates use the usual lock_write, lock_read and unlock mechanisms.
    Unlike most bzr disk formats, DirStates must be locked for reading, using
    lock_read.  (This is an os file lock internally.)  This is necessary
    because the file can be rewritten in place.

    DirStates must be explicitly written with save() to commit changes; just
    unlocking them does not write the changes to disk.
    """
    _kind_to_minikind = {'absent': b'a', 'file': b'f', 'directory': b'd', 'relocated': b'r', 'symlink': b'l', 'tree-reference': b't'}
    _minikind_to_kind = {b'a': 'absent', b'f': 'file', b'd': 'directory', b'l': 'symlink', b'r': 'relocated', b't': 'tree-reference'}
    _stat_to_minikind = {stat.S_IFDIR: b'd', stat.S_IFREG: b'f', stat.S_IFLNK: b'l'}
    _to_yesno = {True: b'y', False: b'n'}
    BISECT_PAGE_SIZE = 4096
    NOT_IN_MEMORY = 0
    IN_MEMORY_UNMODIFIED = 1
    IN_MEMORY_MODIFIED = 2
    IN_MEMORY_HASH_MODIFIED = 3
    NULLSTAT = b'x' * 32
    NULL_PARENT_DETAILS = static_tuple.StaticTuple(b'a', b'', 0, False, b'')
    HEADER_FORMAT_2 = b'#bazaar dirstate flat format 2\n'
    HEADER_FORMAT_3 = b'#bazaar dirstate flat format 3\n'

    def __init__(self, path, sha1_provider, worth_saving_limit=0, use_filesystem_for_exec=True):
        """Create a  DirState object.

        :param path: The path at which the dirstate file on disk should live.
        :param sha1_provider: an object meeting the SHA1Provider interface.
        :param worth_saving_limit: when the exact number of hash changed
            entries is known, only bother saving the dirstate if more than
            this count of entries have changed.
            -1 means never save hash changes, 0 means always save hash changes.
        :param use_filesystem_for_exec: Whether to trust the filesystem
            for executable bit information
        """
        self._header_state = DirState.NOT_IN_MEMORY
        self._dirblock_state = DirState.NOT_IN_MEMORY
        self._changes_aborted = False
        self._dirblocks = []
        self._ghosts = []
        self._parents = []
        self._state_file = None
        self._filename = path
        self._lock_token = None
        self._lock_state = None
        self._id_index = None
        self._packed_stat_index = None
        self._end_of_header = None
        self._cutoff_time = None
        self._split_path_cache = {}
        self._bisect_page_size = DirState.BISECT_PAGE_SIZE
        self._sha1_provider = sha1_provider
        if 'hashcache' in debug.debug_flags:
            self._sha1_file = self._sha1_file_and_mutter
        else:
            self._sha1_file = self._sha1_provider.sha1
        self._last_block_index = None
        self._last_entry_index = None
        self._known_hash_changes = set()
        self._worth_saving_limit = worth_saving_limit
        self._config_stack = config.LocationStack(urlutils.local_path_to_url(path))
        self._use_filesystem_for_exec = use_filesystem_for_exec

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._filename)

    def _mark_modified(self, hash_changed_entries=None, header_modified=False):
        """Mark this dirstate as modified.

        :param hash_changed_entries: if non-None, mark just these entries as
            having their hash modified.
        :param header_modified: mark the header modified as well, not just the
            dirblocks.
        """
        if hash_changed_entries:
            self._known_hash_changes.update([e[0] for e in hash_changed_entries])
            if self._dirblock_state in (DirState.NOT_IN_MEMORY, DirState.IN_MEMORY_UNMODIFIED):
                self._dirblock_state = DirState.IN_MEMORY_HASH_MODIFIED
        else:
            self._dirblock_state = DirState.IN_MEMORY_MODIFIED
        if header_modified:
            self._header_state = DirState.IN_MEMORY_MODIFIED

    def _mark_unmodified(self):
        """Mark this dirstate as unmodified."""
        self._header_state = DirState.IN_MEMORY_UNMODIFIED
        self._dirblock_state = DirState.IN_MEMORY_UNMODIFIED
        self._known_hash_changes = set()

    def add(self, path, file_id, kind, stat, fingerprint):
        """Add a path to be tracked.

        :param path: The path within the dirstate - b'' is the root, 'foo' is the
            path foo within the root, 'foo/bar' is the path bar within foo
            within the root.
        :param file_id: The file id of the path being added.
        :param kind: The kind of the path, as a string like 'file',
            'directory', etc.
        :param stat: The output of os.lstat for the path.
        :param fingerprint: The sha value of the file's canonical form (i.e.
            after any read filters have been applied),
            or the target of a symlink,
            or the referenced revision id for tree-references,
            or b'' for directories.
        """
        dirname, basename = osutils.split(path)
        norm_name, can_access = osutils.normalized_filename(basename)
        if norm_name != basename:
            if can_access:
                basename = norm_name
            else:
                raise errors.InvalidNormalization(path)
        if basename == '.' or basename == '..':
            raise inventory.InvalidEntryName(path)
        utf8path = (dirname + '/' + basename).strip('/').encode('utf8')
        dirname, basename = osutils.split(utf8path)
        if file_id.__class__ is not bytes:
            raise AssertionError('must be a utf8 file_id not {}'.format(type(file_id)))
        rename_from = None
        file_id_entry = self._get_entry(0, fileid_utf8=file_id, include_deleted=True)
        if file_id_entry != (None, None):
            if file_id_entry[1][0][0] == b'a':
                if file_id_entry[0] != (dirname, basename, file_id):
                    self.update_minimal(file_id_entry[0], b'r', path_utf8=b'', packed_stat=b'', fingerprint=utf8path)
                    rename_from = file_id_entry[0][0:2]
            else:
                path = osutils.pathjoin(file_id_entry[0][0], file_id_entry[0][1])
                kind = DirState._minikind_to_kind[file_id_entry[1][0][0]]
                info = '{}:{}'.format(kind, path)
                raise inventory.DuplicateFileId(file_id, info)
        first_key = (dirname, basename, b'')
        block_index, present = self._find_block_index_from_key(first_key)
        if present:
            block = self._dirblocks[block_index][1]
            entry_index, _ = self._find_entry_index(first_key, block)
            while entry_index < len(block) and block[entry_index][0][0:2] == first_key[0:2]:
                if block[entry_index][1][0][0] not in (b'a', b'r'):
                    raise Exception('adding already added path!')
                entry_index += 1
        else:
            parent_dir, parent_base = osutils.split(dirname)
            parent_block_idx, parent_entry_idx, _, parent_present = self._get_block_entry_index(parent_dir, parent_base, 0)
            if not parent_present:
                raise errors.NotVersionedError(path, str(self))
            self._ensure_block(parent_block_idx, parent_entry_idx, dirname)
        block = self._dirblocks[block_index][1]
        entry_key = (dirname, basename, file_id)
        if stat is None:
            size = 0
            packed_stat = DirState.NULLSTAT
        else:
            size = stat.st_size
            packed_stat = pack_stat(stat)
        parent_info = self._empty_parent_info()
        minikind = DirState._kind_to_minikind[kind]
        if rename_from is not None:
            if rename_from[0]:
                old_path_utf8 = b'%s/%s' % rename_from
            else:
                old_path_utf8 = rename_from[1]
            parent_info[0] = (b'r', old_path_utf8, 0, False, b'')
        if kind == 'file':
            entry_data = (entry_key, [(minikind, fingerprint, size, False, packed_stat)] + parent_info)
        elif kind == 'directory':
            entry_data = (entry_key, [(minikind, b'', 0, False, packed_stat)] + parent_info)
        elif kind == 'symlink':
            entry_data = (entry_key, [(minikind, fingerprint, size, False, packed_stat)] + parent_info)
        elif kind == 'tree-reference':
            entry_data = (entry_key, [(minikind, fingerprint, 0, False, packed_stat)] + parent_info)
        else:
            raise errors.BzrError('unknown kind %r' % kind)
        entry_index, present = self._find_entry_index(entry_key, block)
        if not present:
            block.insert(entry_index, entry_data)
        else:
            if block[entry_index][1][0][0] != b'a':
                raise AssertionError(' %r(%r) already added' % (basename, file_id))
            block[entry_index][1][0] = entry_data[1][0]
        if kind == 'directory':
            self._ensure_block(block_index, entry_index, utf8path)
        self._mark_modified()
        if self._id_index:
            self._add_to_id_index(self._id_index, entry_key)

    def _bisect(self, paths):
        """Bisect through the disk structure for specific rows.

        :param paths: A list of paths to find
        :return: A dict mapping path => entries for found entries. Missing
                 entries will not be in the map.
                 The list is not sorted, and entries will be populated
                 based on when they were read.
        """
        self._requires_lock()
        self._read_header_if_needed()
        if self._dirblock_state != DirState.NOT_IN_MEMORY:
            raise AssertionError('bad dirblock state %r' % self._dirblock_state)
        state_file = self._state_file
        file_size = os.fstat(state_file.fileno()).st_size
        entry_field_count = self._fields_per_entry() + 1
        low = self._end_of_header
        high = file_size - 1
        found = {}
        max_count = 30 * len(paths)
        count = 0
        pending = [(low, high, paths)]
        page_size = self._bisect_page_size
        fields_to_entry = self._get_fields_to_entry()
        while pending:
            low, high, cur_files = pending.pop()
            if not cur_files or low >= high:
                continue
            count += 1
            if count > max_count:
                raise errors.BzrError('Too many seeks, most likely a bug.')
            mid = max(low, (low + high - page_size) // 2)
            state_file.seek(mid)
            read_size = min(page_size, high - mid + 1)
            block = state_file.read(read_size)
            start = mid
            entries = block.split(b'\n')
            if len(entries) < 2:
                page_size *= 2
                pending.append((low, high, cur_files))
                continue
            first_entry_num = 0
            first_fields = entries[0].split(b'\x00')
            if len(first_fields) < entry_field_count:
                start += len(entries[0]) + 1
                first_fields = entries[1].split(b'\x00')
                first_entry_num = 1
            if len(first_fields) <= 2:
                page_size *= 2
                pending.append((low, high, cur_files))
                continue
            else:
                after = start
                if first_fields[1]:
                    first_path = first_fields[1] + b'/' + first_fields[2]
                else:
                    first_path = first_fields[2]
                first_loc = _bisect_path_left(cur_files, first_path)
                pre = cur_files[:first_loc]
                post = cur_files[first_loc:]
            if post and len(first_fields) >= entry_field_count:
                last_entry_num = len(entries) - 1
                last_fields = entries[last_entry_num].split(b'\x00')
                if len(last_fields) < entry_field_count:
                    after = mid + len(block) - len(entries[-1])
                    last_entry_num -= 1
                    last_fields = entries[last_entry_num].split(b'\x00')
                else:
                    after = mid + len(block)
                if last_fields[1]:
                    last_path = last_fields[1] + b'/' + last_fields[2]
                else:
                    last_path = last_fields[2]
                last_loc = _bisect_path_right(post, last_path)
                middle_files = post[:last_loc]
                post = post[last_loc:]
                if middle_files:
                    if middle_files[0] == first_path:
                        pre.append(first_path)
                    if middle_files[-1] == last_path:
                        post.insert(0, last_path)
                    paths = {first_path: [first_fields]}
                    if last_entry_num != first_entry_num:
                        paths.setdefault(last_path, []).append(last_fields)
                    for num in range(first_entry_num + 1, last_entry_num):
                        fields = entries[num].split(b'\x00')
                        if fields[1]:
                            path = fields[1] + b'/' + fields[2]
                        else:
                            path = fields[2]
                        paths.setdefault(path, []).append(fields)
                    for path in middle_files:
                        for fields in paths.get(path, []):
                            entry = fields_to_entry(fields[1:])
                            found.setdefault(path, []).append(entry)
            if post:
                pending.append((after, high, post))
            if pre:
                pending.append((low, start - 1, pre))
        return found

    def _bisect_dirblocks(self, dir_list):
        """Bisect through the disk structure to find entries in given dirs.

        _bisect_dirblocks is meant to find the contents of directories, which
        differs from _bisect, which only finds individual entries.

        :param dir_list: A sorted list of directory names ['', 'dir', 'foo'].
        :return: A map from dir => entries_for_dir
        """
        self._requires_lock()
        self._read_header_if_needed()
        if self._dirblock_state != DirState.NOT_IN_MEMORY:
            raise AssertionError('bad dirblock state %r' % self._dirblock_state)
        state_file = self._state_file
        file_size = os.fstat(state_file.fileno()).st_size
        entry_field_count = self._fields_per_entry() + 1
        low = self._end_of_header
        high = file_size - 1
        found = {}
        max_count = 30 * len(dir_list)
        count = 0
        pending = [(low, high, dir_list)]
        page_size = self._bisect_page_size
        fields_to_entry = self._get_fields_to_entry()
        while pending:
            low, high, cur_dirs = pending.pop()
            if not cur_dirs or low >= high:
                continue
            count += 1
            if count > max_count:
                raise errors.BzrError('Too many seeks, most likely a bug.')
            mid = max(low, (low + high - page_size) // 2)
            state_file.seek(mid)
            read_size = min(page_size, high - mid + 1)
            block = state_file.read(read_size)
            start = mid
            entries = block.split(b'\n')
            if len(entries) < 2:
                page_size *= 2
                pending.append((low, high, cur_dirs))
                continue
            first_entry_num = 0
            first_fields = entries[0].split(b'\x00')
            if len(first_fields) < entry_field_count:
                start += len(entries[0]) + 1
                first_fields = entries[1].split(b'\x00')
                first_entry_num = 1
            if len(first_fields) <= 1:
                page_size *= 2
                pending.append((low, high, cur_dirs))
                continue
            else:
                after = start
                first_dir = first_fields[1]
                first_loc = bisect.bisect_left(cur_dirs, first_dir)
                pre = cur_dirs[:first_loc]
                post = cur_dirs[first_loc:]
            if post and len(first_fields) >= entry_field_count:
                last_entry_num = len(entries) - 1
                last_fields = entries[last_entry_num].split(b'\x00')
                if len(last_fields) < entry_field_count:
                    after = mid + len(block) - len(entries[-1])
                    last_entry_num -= 1
                    last_fields = entries[last_entry_num].split(b'\x00')
                else:
                    after = mid + len(block)
                last_dir = last_fields[1]
                last_loc = bisect.bisect_right(post, last_dir)
                middle_files = post[:last_loc]
                post = post[last_loc:]
                if middle_files:
                    if middle_files[0] == first_dir:
                        pre.append(first_dir)
                    if middle_files[-1] == last_dir:
                        post.insert(0, last_dir)
                    paths = {first_dir: [first_fields]}
                    if last_entry_num != first_entry_num:
                        paths.setdefault(last_dir, []).append(last_fields)
                    for num in range(first_entry_num + 1, last_entry_num):
                        fields = entries[num].split(b'\x00')
                        paths.setdefault(fields[1], []).append(fields)
                    for cur_dir in middle_files:
                        for fields in paths.get(cur_dir, []):
                            entry = fields_to_entry(fields[1:])
                            found.setdefault(cur_dir, []).append(entry)
            if post:
                pending.append((after, high, post))
            if pre:
                pending.append((low, start - 1, pre))
        return found

    def _bisect_recursive(self, paths):
        """Bisect for entries for all paths and their children.

        This will use bisect to find all records for the supplied paths. It
        will then continue to bisect for any records which are marked as
        directories. (and renames?)

        :param paths: A sorted list of (dir, name) pairs
             eg: [('', b'a'), ('', b'f'), ('a/b', b'c')]
        :return: A dictionary mapping (dir, name, file_id) => [tree_info]
        """
        found = {}
        found_dir_names = set()
        processed_dirs = set()
        newly_found = self._bisect(paths)
        while newly_found:
            pending_dirs = set()
            paths_to_search = set()
            for entry_list in newly_found.values():
                for dir_name_id, trees_info in entry_list:
                    found[dir_name_id] = trees_info
                    found_dir_names.add(dir_name_id[:2])
                    is_dir = False
                    for tree_info in trees_info:
                        minikind = tree_info[0]
                        if minikind == b'd':
                            if is_dir:
                                continue
                            subdir, name, file_id = dir_name_id
                            path = osutils.pathjoin(subdir, name)
                            is_dir = True
                            if path not in processed_dirs:
                                pending_dirs.add(path)
                        elif minikind == b'r':
                            dir_name = osutils.split(tree_info[1])
                            if dir_name[0] in pending_dirs:
                                continue
                            if dir_name not in found_dir_names:
                                paths_to_search.add(tree_info[1])
            newly_found = self._bisect(sorted(paths_to_search))
            newly_found.update(self._bisect_dirblocks(sorted(pending_dirs)))
            processed_dirs.update(pending_dirs)
        return found

    def _discard_merge_parents(self):
        """Discard any parents trees beyond the first.

        Note that if this fails the dirstate is corrupted.

        After this function returns the dirstate contains 2 trees, neither of
        which are ghosted.
        """
        self._read_header_if_needed()
        parents = self.get_parent_ids()
        if len(parents) < 1:
            return
        self._read_dirblocks_if_needed()
        dead_patterns = {(b'a', b'r'), (b'a', b'a'), (b'r', b'r'), (b'r', b'a')}

        def iter_entries_removable():
            for block in self._dirblocks:
                deleted_positions = []
                for pos, entry in enumerate(block[1]):
                    yield entry
                    if (entry[1][0][0], entry[1][1][0]) in dead_patterns:
                        deleted_positions.append(pos)
                if deleted_positions:
                    if len(deleted_positions) == len(block[1]):
                        del block[1][:]
                    else:
                        for pos in reversed(deleted_positions):
                            del block[1][pos]
        if parents[0] in self.get_ghosts():
            empty_parent = [DirState.NULL_PARENT_DETAILS]
            for entry in iter_entries_removable():
                entry[1][1:] = empty_parent
        else:
            for entry in iter_entries_removable():
                del entry[1][2:]
        self._ghosts = []
        self._parents = [parents[0]]
        self._mark_modified(header_modified=True)

    def _empty_parent_info(self):
        return [DirState.NULL_PARENT_DETAILS] * (len(self._parents) - len(self._ghosts))

    def _ensure_block(self, parent_block_index, parent_row_index, dirname):
        """Ensure a block for dirname exists.

        This function exists to let callers which know that there is a
        directory dirname ensure that the block for it exists. This block can
        fail to exist because of demand loading, or because a directory had no
        children. In either case it is not an error. It is however an error to
        call this if there is no parent entry for the directory, and thus the
        function requires the coordinates of such an entry to be provided.

        The root row is special cased and can be indicated with a parent block
        and row index of -1

        :param parent_block_index: The index of the block in which dirname's row
            exists.
        :param parent_row_index: The index in the parent block where the row
            exists.
        :param dirname: The utf8 dirname to ensure there is a block for.
        :return: The index for the block.
        """
        if dirname == b'' and parent_row_index == 0 and (parent_block_index == 0):
            return 1
        if not (parent_block_index == -1 and parent_block_index == -1 and (dirname == b'')):
            if not dirname.endswith(self._dirblocks[parent_block_index][1][parent_row_index][0][1]):
                raise AssertionError('bad dirname %r' % dirname)
        block_index, present = self._find_block_index_from_key((dirname, b'', b''))
        if not present:
            self._dirblocks.insert(block_index, (dirname, []))
        return block_index

    def _entries_to_current_state(self, new_entries):
        """Load new_entries into self.dirblocks.

        Process new_entries into the current state object, making them the active
        state.  The entries are grouped together by directory to form dirblocks.

        :param new_entries: A sorted list of entries. This function does not sort
            to prevent unneeded overhead when callers have a sorted list already.
        :return: Nothing.
        """
        if new_entries[0][0][0:2] != (b'', b''):
            raise AssertionError('Missing root row {!r}'.format(new_entries[0][0]))
        self._dirblocks = [(b'', []), (b'', [])]
        current_block = self._dirblocks[0][1]
        current_dirname = b''
        root_key = (b'', b'')
        append_entry = current_block.append
        for entry in new_entries:
            if entry[0][0] != current_dirname:
                current_block = []
                current_dirname = entry[0][0]
                self._dirblocks.append((current_dirname, current_block))
                append_entry = current_block.append
            append_entry(entry)
        self._split_root_dirblock_into_contents()

    def _split_root_dirblock_into_contents(self):
        """Split the root dirblocks into root and contents-of-root.

        After parsing by path, we end up with root entries and contents-of-root
        entries in the same block. This loop splits them out again.
        """
        if self._dirblocks[1] != (b'', []):
            raise ValueError('bad dirblock start {!r}'.format(self._dirblocks[1]))
        root_block = []
        contents_of_root_block = []
        for entry in self._dirblocks[0][1]:
            if not entry[0][1]:
                root_block.append(entry)
            else:
                contents_of_root_block.append(entry)
        self._dirblocks[0] = (b'', root_block)
        self._dirblocks[1] = (b'', contents_of_root_block)

    def _entries_for_path(self, path):
        """Return a list with all the entries that match path for all ids."""
        dirname, basename = os.path.split(path)
        key = (dirname, basename, b'')
        block_index, present = self._find_block_index_from_key(key)
        if not present:
            return []
        result = []
        block = self._dirblocks[block_index][1]
        entry_index, _ = self._find_entry_index(key, block)
        while entry_index < len(block) and block[entry_index][0][0:2] == key[0:2]:
            result.append(block[entry_index])
            entry_index += 1
        return result

    def _entry_to_line(self, entry):
        """Serialize entry to a NULL delimited line ready for _get_output_lines.

        :param entry: An entry_tuple as defined in the module docstring.
        """
        entire_entry = list(entry[0])
        for tree_number, tree_data in enumerate(entry[1]):
            entire_entry.extend(tree_data)
            tree_offset = 3 + tree_number * 5
            entire_entry[tree_offset + 0] = tree_data[0]
            entire_entry[tree_offset + 2] = b'%d' % tree_data[2]
            entire_entry[tree_offset + 3] = DirState._to_yesno[tree_data[3]]
        return b'\x00'.join(entire_entry)

    def _fields_per_entry(self):
        """How many null separated fields should be in each entry row.

        Each line now has an extra '\\n' field which is not used
        so we just skip over it

        entry size::
            3 fields for the key
            + number of fields per tree_data (5) * tree count
            + newline
         """
        tree_count = 1 + self._num_present_parents()
        return 3 + 5 * tree_count + 1

    def _find_block(self, key, add_if_missing=False):
        """Return the block that key should be present in.

        :param key: A dirstate entry key.
        :return: The block tuple.
        """
        block_index, present = self._find_block_index_from_key(key)
        if not present:
            if not add_if_missing:
                parent_base, parent_name = osutils.split(key[0])
                if not self._get_block_entry_index(parent_base, parent_name, 0)[3]:
                    raise errors.NotVersionedError(key[0:2], str(self))
            self._dirblocks.insert(block_index, (key[0], []))
        return self._dirblocks[block_index]

    def _find_block_index_from_key(self, key):
        """Find the dirblock index for a key.

        :return: The block index, True if the block for the key is present.
        """
        if key[0:2] == (b'', b''):
            return (0, True)
        try:
            if self._last_block_index is not None and self._dirblocks[self._last_block_index][0] == key[0]:
                return (self._last_block_index, True)
        except IndexError:
            pass
        block_index = bisect_dirblock(self._dirblocks, key[0], 1, cache=self._split_path_cache)
        present = block_index < len(self._dirblocks) and self._dirblocks[block_index][0] == key[0]
        self._last_block_index = block_index
        self._last_entry_index = -1
        return (block_index, present)

    def _find_entry_index(self, key, block):
        """Find the entry index for a key in a block.

        :return: The entry index, True if the entry for the key is present.
        """
        len_block = len(block)
        try:
            if self._last_entry_index is not None:
                entry_index = self._last_entry_index + 1
                if (entry_index > 0 and block[entry_index - 1][0] < key) and key <= block[entry_index][0]:
                    self._last_entry_index = entry_index
                    present = block[entry_index][0] == key
                    return (entry_index, present)
        except IndexError:
            pass
        entry_index = bisect.bisect_left(block, (key, []))
        present = entry_index < len_block and block[entry_index][0] == key
        self._last_entry_index = entry_index
        return (entry_index, present)

    @staticmethod
    def from_tree(tree, dir_state_filename, sha1_provider=None):
        """Create a dirstate from a bzr Tree.

        :param tree: The tree which should provide parent information and
            inventory ids.
        :param sha1_provider: an object meeting the SHA1Provider interface.
            If None, a DefaultSHA1Provider is used.
        :return: a DirState object which is currently locked for writing.
            (it was locked by DirState.initialize)
        """
        result = DirState.initialize(dir_state_filename, sha1_provider=sha1_provider)
        try:
            with contextlib.ExitStack() as exit_stack:
                exit_stack.enter_context(tree.lock_read())
                parent_ids = tree.get_parent_ids()
                num_parents = len(parent_ids)
                parent_trees = []
                for parent_id in parent_ids:
                    parent_tree = tree.branch.repository.revision_tree(parent_id)
                    parent_trees.append((parent_id, parent_tree))
                    exit_stack.enter_context(parent_tree.lock_read())
                result.set_parent_trees(parent_trees, [])
                result.set_state_from_inventory(tree.root_inventory)
        except:
            result.unlock()
            raise
        return result

    def _check_delta_is_valid(self, delta):
        delta = list(inventory._check_delta_unique_ids(inventory._check_delta_unique_old_paths(inventory._check_delta_unique_new_paths(inventory._check_delta_ids_match_entry(inventory._check_delta_ids_are_valid(inventory._check_delta_new_path_entry_both_or_None(delta)))))))

        def delta_key(d):
            old_path, new_path, file_id, new_entry = d
            if old_path is None:
                old_path = ''
            if new_path is None:
                new_path = ''
            return (old_path, new_path, file_id, new_entry)
        delta.sort(key=delta_key, reverse=True)
        return delta

    def update_by_delta(self, delta):
        """Apply an inventory delta to the dirstate for tree 0

        This is the workhorse for apply_inventory_delta in dirstate based
        trees.

        :param delta: An inventory delta.  See Inventory.apply_delta for
            details.
        """
        self._read_dirblocks_if_needed()
        encode = cache_utf8.encode
        insertions = {}
        removals = {}
        parents = set()
        new_ids = set()
        delta = self._check_delta_is_valid(delta)
        for old_path, new_path, file_id, inv_entry in delta:
            if not isinstance(file_id, bytes):
                raise AssertionError('must be a utf8 file_id not {}'.format(type(file_id)))
            if file_id in insertions or file_id in removals:
                self._raise_invalid(old_path or new_path, file_id, 'repeated file_id')
            if old_path is not None:
                old_path = old_path.encode('utf-8')
                removals[file_id] = old_path
            else:
                new_ids.add(file_id)
            if new_path is not None:
                if inv_entry is None:
                    self._raise_invalid(new_path, file_id, 'new_path with no entry')
                new_path = new_path.encode('utf-8')
                dirname_utf8, basename = osutils.split(new_path)
                if basename:
                    parents.add((dirname_utf8, inv_entry.parent_id))
                key = (dirname_utf8, basename, file_id)
                minikind = DirState._kind_to_minikind[inv_entry.kind]
                if minikind == b't':
                    fingerprint = inv_entry.reference_revision or b''
                else:
                    fingerprint = b''
                insertions[file_id] = (key, minikind, inv_entry.executable, fingerprint, new_path)
            if None not in (old_path, new_path):
                for child in self._iter_child_entries(0, old_path):
                    if child[0][2] in insertions or child[0][2] in removals:
                        continue
                    child_dirname = child[0][0]
                    child_basename = child[0][1]
                    minikind = child[1][0][0]
                    fingerprint = child[1][0][4]
                    executable = child[1][0][3]
                    old_child_path = osutils.pathjoin(child_dirname, child_basename)
                    removals[child[0][2]] = old_child_path
                    child_suffix = child_dirname[len(old_path):]
                    new_child_dirname = new_path + child_suffix
                    key = (new_child_dirname, child_basename, child[0][2])
                    new_child_path = osutils.pathjoin(new_child_dirname, child_basename)
                    insertions[child[0][2]] = (key, minikind, executable, fingerprint, new_child_path)
        self._check_delta_ids_absent(new_ids, delta, 0)
        try:
            self._apply_removals(removals.items())
            self._apply_insertions(insertions.values())
            self._after_delta_check_parents(parents, 0)
        except errors.BzrError as e:
            self._changes_aborted = True
            if 'integrity error' not in str(e):
                raise
            raise errors.InconsistentDeltaDelta(delta, 'error from _get_entry. {}'.format(e))

    def _apply_removals(self, removals):
        for file_id, path in sorted(removals, reverse=True, key=operator.itemgetter(1)):
            dirname, basename = osutils.split(path)
            block_i, entry_i, d_present, f_present = self._get_block_entry_index(dirname, basename, 0)
            try:
                entry = self._dirblocks[block_i][1][entry_i]
            except IndexError:
                self._raise_invalid(path, file_id, 'Wrong path for old path.')
            if not f_present or entry[1][0][0] in (b'a', b'r'):
                self._raise_invalid(path, file_id, 'Wrong path for old path.')
            if file_id != entry[0][2]:
                self._raise_invalid(path, file_id, 'Attempt to remove path has wrong id - found %r.' % entry[0][2])
            self._make_absent(entry)
            block_i, entry_i, d_present, f_present = self._get_block_entry_index(path, b'', 0)
            if d_present:
                for child_entry in self._dirblocks[block_i][1]:
                    if child_entry[1][0][0] not in (b'r', b'a'):
                        self._raise_invalid(path, entry[0][2], 'The file id was deleted but its children were not deleted.')

    def _apply_insertions(self, adds):
        try:
            for key, minikind, executable, fingerprint, path_utf8 in sorted(adds):
                self.update_minimal(key, minikind, executable, fingerprint, path_utf8=path_utf8)
        except errors.NotVersionedError:
            self._raise_invalid(path_utf8.decode('utf8'), key[2], 'Missing parent')

    def update_basis_by_delta(self, delta, new_revid):
        """Update the parents of this tree after a commit.

        This gives the tree one parent, with revision id new_revid. The
        inventory delta is applied to the current basis tree to generate the
        inventory for the parent new_revid, and all other parent trees are
        discarded.

        Note that an exception during the operation of this method will leave
        the dirstate in a corrupt state where it should not be saved.

        :param new_revid: The new revision id for the trees parent.
        :param delta: An inventory delta (see apply_inventory_delta) describing
            the changes from the current left most parent revision to new_revid.
        """
        self._read_dirblocks_if_needed()
        self._discard_merge_parents()
        if self._ghosts != []:
            raise NotImplementedError(self.update_basis_by_delta)
        if len(self._parents) == 0:
            empty_parent = DirState.NULL_PARENT_DETAILS
            for entry in self._iter_entries():
                entry[1].append(empty_parent)
            self._parents.append(new_revid)
        self._parents[0] = new_revid
        delta = self._check_delta_is_valid(delta)
        adds = []
        changes = []
        deletes = []
        encode = cache_utf8.encode
        inv_to_entry = self._inv_entry_to_details
        root_only = ('', '')
        parents = set()
        new_ids = set()
        for old_path, new_path, file_id, inv_entry in delta:
            if file_id.__class__ is not bytes:
                raise AssertionError('must be a utf8 file_id not {}'.format(type(file_id)))
            if inv_entry is not None and file_id != inv_entry.file_id:
                self._raise_invalid(new_path, file_id, 'mismatched entry file_id %r' % inv_entry)
            if new_path is None:
                new_path_utf8 = None
            else:
                if inv_entry is None:
                    self._raise_invalid(new_path, file_id, 'new_path with no entry')
                new_path_utf8 = encode(new_path)
                dirname_utf8, basename_utf8 = osutils.split(new_path_utf8)
                if basename_utf8:
                    parents.add((dirname_utf8, inv_entry.parent_id))
            if old_path is None:
                old_path_utf8 = None
            else:
                old_path_utf8 = encode(old_path)
            if old_path is None:
                adds.append((None, new_path_utf8, file_id, inv_to_entry(inv_entry), True))
                new_ids.add(file_id)
            elif new_path is None:
                deletes.append((old_path_utf8, None, file_id, None, True))
            elif (old_path, new_path) == root_only:
                changes.append((old_path_utf8, new_path_utf8, file_id, inv_to_entry(inv_entry)))
            else:
                self._update_basis_apply_deletes(deletes)
                deletes = []
                adds.append((old_path_utf8, new_path_utf8, file_id, inv_to_entry(inv_entry), False))
                new_deletes = reversed(list(self._iter_child_entries(1, old_path_utf8)))
                for entry in new_deletes:
                    child_dirname, child_basename, child_file_id = entry[0]
                    if child_dirname:
                        source_path = child_dirname + b'/' + child_basename
                    else:
                        source_path = child_basename
                    if new_path_utf8:
                        target_path = new_path_utf8 + source_path[len(old_path_utf8):]
                    else:
                        if old_path_utf8 == b'':
                            raise AssertionError('cannot rename directory to itself')
                        target_path = source_path[len(old_path_utf8) + 1:]
                    adds.append((None, target_path, entry[0][2], entry[1][1], False))
                    deletes.append((source_path, target_path, entry[0][2], None, False))
                deletes.append((old_path_utf8, new_path_utf8, file_id, None, False))
        self._check_delta_ids_absent(new_ids, delta, 1)
        try:
            self._update_basis_apply_deletes(deletes)
            self._update_basis_apply_adds(adds)
            self._update_basis_apply_changes(changes)
            self._after_delta_check_parents(parents, 1)
        except errors.BzrError as e:
            self._changes_aborted = True
            if 'integrity error' not in str(e):
                raise
            raise errors.InconsistentDeltaDelta(delta, 'error from _get_entry. {}'.format(e))
        self._mark_modified(header_modified=True)
        self._id_index = None
        return

    def _check_delta_ids_absent(self, new_ids, delta, tree_index):
        """Check that none of the file_ids in new_ids are present in a tree."""
        if not new_ids:
            return
        id_index = self._get_id_index()
        for file_id in new_ids:
            for key in id_index.get(file_id, ()):
                block_i, entry_i, d_present, f_present = self._get_block_entry_index(key[0], key[1], tree_index)
                if not f_present:
                    continue
                entry = self._dirblocks[block_i][1][entry_i]
                if entry[0][2] != file_id:
                    continue
                self._raise_invalid((b'%s/%s' % key[0:2]).decode('utf8'), file_id, 'This file_id is new in the delta but already present in the target')

    def _raise_invalid(self, path, file_id, reason):
        self._changes_aborted = True
        raise errors.InconsistentDelta(path, file_id, reason)

    def _update_basis_apply_adds(self, adds):
        """Apply a sequence of adds to tree 1 during update_basis_by_delta.

        They may be adds, or renames that have been split into add/delete
        pairs.

        :param adds: A sequence of adds. Each add is a tuple:
            (None, new_path_utf8, file_id, (entry_details), real_add). real_add
            is False when the add is the second half of a remove-and-reinsert
            pair created to handle renames and deletes.
        """
        adds.sort(key=lambda x: x[1])
        st = static_tuple.StaticTuple
        for old_path, new_path, file_id, new_details, real_add in adds:
            dirname, basename = osutils.split(new_path)
            entry_key = st(dirname, basename, file_id)
            block_index, present = self._find_block_index_from_key(entry_key)
            if not present:
                parent_dir, parent_base = osutils.split(dirname)
                parent_block_idx, parent_entry_idx, _, parent_present = self._get_block_entry_index(parent_dir, parent_base, 1)
                if not parent_present:
                    self._raise_invalid(new_path, file_id, 'Unable to find block for this record. Was the parent added?')
                self._ensure_block(parent_block_idx, parent_entry_idx, dirname)
            block = self._dirblocks[block_index][1]
            entry_index, present = self._find_entry_index(entry_key, block)
            if real_add:
                if old_path is not None:
                    self._raise_invalid(new_path, file_id, 'considered a real add but still had old_path at %s' % (old_path,))
            if present:
                entry = block[entry_index]
                basis_kind = entry[1][1][0]
                if basis_kind == b'a':
                    entry[1][1] = new_details
                elif basis_kind == b'r':
                    raise NotImplementedError()
                else:
                    self._raise_invalid(new_path, file_id, 'An entry was marked as a new add but the basis target already existed')
            else:
                for maybe_index in range(entry_index - 1, entry_index + 1):
                    if maybe_index < 0 or maybe_index >= len(block):
                        continue
                    maybe_entry = block[maybe_index]
                    if maybe_entry[0][:2] != (dirname, basename):
                        continue
                    if maybe_entry[0][2] == file_id:
                        raise AssertionError('_find_entry_index didnt find a key match but walking the data did, for %s' % (entry_key,))
                    basis_kind = maybe_entry[1][1][0]
                    if basis_kind not in (b'a', b'r'):
                        self._raise_invalid(new_path, file_id, 'we have an add record for path, but the path is already present with another file_id %s' % (maybe_entry[0][2],))
                entry = (entry_key, [DirState.NULL_PARENT_DETAILS, new_details])
                block.insert(entry_index, entry)
            active_kind = entry[1][0][0]
            if active_kind == b'a':
                id_index = self._get_id_index()
                keys = id_index.get(file_id, ())
                for key in keys:
                    block_i, entry_i, d_present, f_present = self._get_block_entry_index(key[0], key[1], 0)
                    if not f_present:
                        continue
                    active_entry = self._dirblocks[block_i][1][entry_i]
                    if active_entry[0][2] != file_id:
                        continue
                    real_active_kind = active_entry[1][0][0]
                    if real_active_kind in (b'a', b'r'):
                        self._raise_invalid(new_path, file_id, 'We found a tree0 entry that doesnt make sense')
                    active_dir, active_name = active_entry[0][:2]
                    if active_dir:
                        active_path = active_dir + b'/' + active_name
                    else:
                        active_path = active_name
                    active_entry[1][1] = st(b'r', new_path, 0, False, b'')
                    entry[1][0] = st(b'r', active_path, 0, False, b'')
            elif active_kind == b'r':
                raise NotImplementedError()
            new_kind = new_details[0]
            if new_kind == b'd':
                self._ensure_block(block_index, entry_index, new_path)

    def _update_basis_apply_changes(self, changes):
        """Apply a sequence of changes to tree 1 during update_basis_by_delta.

        :param adds: A sequence of changes. Each change is a tuple:
            (path_utf8, path_utf8, file_id, (entry_details))
        """
        for old_path, new_path, file_id, new_details in changes:
            entry = self._get_entry(1, file_id, new_path)
            if entry[0] is None or entry[1][1][0] in (b'a', b'r'):
                self._raise_invalid(new_path, file_id, 'changed entry considered not present')
            entry[1][1] = new_details

    def _update_basis_apply_deletes(self, deletes):
        """Apply a sequence of deletes to tree 1 during update_basis_by_delta.

        They may be deletes, or renames that have been split into add/delete
        pairs.

        :param deletes: A sequence of deletes. Each delete is a tuple:
            (old_path_utf8, new_path_utf8, file_id, None, real_delete).
            real_delete is True when the desired outcome is an actual deletion
            rather than the rename handling logic temporarily deleting a path
            during the replacement of a parent.
        """
        null = DirState.NULL_PARENT_DETAILS
        for old_path, new_path, file_id, _, real_delete in deletes:
            if real_delete != (new_path is None):
                self._raise_invalid(old_path, file_id, 'bad delete delta')
            dirname, basename = osutils.split(old_path)
            block_index, entry_index, dir_present, file_present = self._get_block_entry_index(dirname, basename, 1)
            if not file_present:
                self._raise_invalid(old_path, file_id, 'basis tree does not contain removed entry')
            entry = self._dirblocks[block_index][1][entry_index]
            active_kind = entry[1][0][0]
            if entry[0][2] != file_id:
                self._raise_invalid(old_path, file_id, 'mismatched file_id in tree 1')
            dir_block = ()
            old_kind = entry[1][1][0]
            if active_kind in b'ar':
                if active_kind == b'r':
                    active_path = entry[1][0][1]
                    active_entry = self._get_entry(0, file_id, active_path)
                    if active_entry[1][1][0] != b'r':
                        self._raise_invalid(old_path, file_id, 'Dirstate did not have matching rename entries')
                    elif active_entry[1][0][0] in b'ar':
                        self._raise_invalid(old_path, file_id, 'Dirstate had a rename pointing at an inactive tree0')
                    active_entry[1][1] = null
                del self._dirblocks[block_index][1][entry_index]
                if old_kind == b'd':
                    dir_block_index, present = self._find_block_index_from_key((old_path, b'', b''))
                    if present:
                        dir_block = self._dirblocks[dir_block_index][1]
                        if not dir_block:
                            del self._dirblocks[dir_block_index]
            else:
                entry[1][1] = null
                block_i, entry_i, d_present, f_present = self._get_block_entry_index(old_path, b'', 1)
                if d_present:
                    dir_block = self._dirblocks[block_i][1]
            for child_entry in dir_block:
                child_basis_kind = child_entry[1][1][0]
                if child_basis_kind not in b'ar':
                    self._raise_invalid(old_path, file_id, 'The file id was deleted but its children were not deleted.')

    def _after_delta_check_parents(self, parents, index):
        """Check that parents required by the delta are all intact.

        :param parents: An iterable of (path_utf8, file_id) tuples which are
            required to be present in tree 'index' at path_utf8 with id file_id
            and be a directory.
        :param index: The column in the dirstate to check for parents in.
        """
        for dirname_utf8, file_id in parents:
            entry = self._get_entry(index, file_id, dirname_utf8)
            if entry[1] is None:
                self._raise_invalid(dirname_utf8.decode('utf8'), file_id, 'This parent is not present.')
            if entry[1][index][0] != b'd':
                self._raise_invalid(dirname_utf8.decode('utf8'), file_id, 'This parent is not a directory.')

    def _observed_sha1(self, entry, sha1, stat_value, _stat_to_minikind=_stat_to_minikind):
        """Note the sha1 of a file.

        :param entry: The entry the sha1 is for.
        :param sha1: The observed sha1.
        :param stat_value: The os.lstat for the file.
        """
        try:
            minikind = _stat_to_minikind[stat_value.st_mode & 61440]
        except KeyError:
            return None
        if minikind == b'f':
            if self._cutoff_time is None:
                self._sha_cutoff_time()
            if stat_value.st_mtime < self._cutoff_time and stat_value.st_ctime < self._cutoff_time:
                entry[1][0] = (b'f', sha1, stat_value.st_size, entry[1][0][3], pack_stat(stat_value))
                self._mark_modified([entry])

    def _sha_cutoff_time(self):
        """Return cutoff time.

        Files modified more recently than this time are at risk of being
        undetectably modified and so can't be cached.
        """
        self._cutoff_time = int(time.time()) - 3
        return self._cutoff_time

    def _lstat(self, abspath, entry):
        """Return the os.lstat value for this path."""
        return os.lstat(abspath)

    def _sha1_file_and_mutter(self, abspath):
        trace.mutter('dirstate sha1 ' + abspath)
        return self._sha1_provider.sha1(abspath)

    def _is_executable(self, mode, old_executable):
        """Is this file executable?"""
        if self._use_filesystem_for_exec:
            return bool(S_IEXEC & mode)
        else:
            return old_executable

    def _read_link(self, abspath, old_link):
        """Read the target of a symlink"""
        if isinstance(abspath, str):
            abspath = os.fsencode(abspath)
        target = os.readlink(abspath)
        if sys.getfilesystemencoding() not in ('utf-8', 'ascii'):
            target = os.fsdecode(target).encode('UTF-8')
        return target

    def get_ghosts(self):
        """Return a list of the parent tree revision ids that are ghosts."""
        self._read_header_if_needed()
        return self._ghosts

    def get_lines(self):
        """Serialise the entire dirstate to a sequence of lines."""
        if self._header_state == DirState.IN_MEMORY_UNMODIFIED and self._dirblock_state == DirState.IN_MEMORY_UNMODIFIED:
            self._state_file.seek(0)
            return self._state_file.readlines()
        lines = []
        lines.append(self._get_parents_line(self.get_parent_ids()))
        lines.append(self._get_ghosts_line(self._ghosts))
        lines.extend(self._iter_entry_lines())
        return self._get_output_lines(lines)

    def _get_ghosts_line(self, ghost_ids):
        """Create a line for the state file for ghost information."""
        return b'\x00'.join([b'%d' % len(ghost_ids)] + ghost_ids)

    def _get_parents_line(self, parent_ids):
        """Create a line for the state file for parents information."""
        return b'\x00'.join([b'%d' % len(parent_ids)] + parent_ids)

    def _iter_entry_lines(self):
        """Create lines for entries."""
        return map(self._entry_to_line, self._iter_entries())

    def _get_fields_to_entry(self):
        """Get a function which converts entry fields into a entry record.

        This handles size and executable, as well as parent records.

        :return: A function which takes a list of fields, and returns an
            appropriate record for storing in memory.
        """
        num_present_parents = self._num_present_parents()
        if num_present_parents == 0:

            def fields_to_entry_0_parents(fields, _int=int):
                path_name_file_id_key = (fields[0], fields[1], fields[2])
                return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7])])
            return fields_to_entry_0_parents
        elif num_present_parents == 1:

            def fields_to_entry_1_parent(fields, _int=int):
                path_name_file_id_key = (fields[0], fields[1], fields[2])
                return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7]), (fields[8], fields[9], _int(fields[10]), fields[11] == b'y', fields[12])])
            return fields_to_entry_1_parent
        elif num_present_parents == 2:

            def fields_to_entry_2_parents(fields, _int=int):
                path_name_file_id_key = (fields[0], fields[1], fields[2])
                return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7]), (fields[8], fields[9], _int(fields[10]), fields[11] == b'y', fields[12]), (fields[13], fields[14], _int(fields[15]), fields[16] == b'y', fields[17])])
            return fields_to_entry_2_parents
        else:

            def fields_to_entry_n_parents(fields, _int=int):
                path_name_file_id_key = (fields[0], fields[1], fields[2])
                trees = [(fields[cur], fields[cur + 1], _int(fields[cur + 2]), fields[cur + 3] == b'y', fields[cur + 4]) for cur in range(3, len(fields) - 1, 5)]
                return (path_name_file_id_key, trees)
            return fields_to_entry_n_parents

    def get_parent_ids(self):
        """Return a list of the parent tree ids for the directory state."""
        self._read_header_if_needed()
        return list(self._parents)

    def _get_block_entry_index(self, dirname, basename, tree_index):
        """Get the coordinates for a path in the state structure.

        :param dirname: The utf8 dirname to lookup.
        :param basename: The utf8 basename to lookup.
        :param tree_index: The index of the tree for which this lookup should
            be attempted.
        :return: A tuple describing where the path is located, or should be
            inserted. The tuple contains four fields: the block index, the row
            index, the directory is present (boolean), the entire path is
            present (boolean).  There is no guarantee that either
            coordinate is currently reachable unless the found field for it is
            True. For instance, a directory not present in the searched tree
            may be returned with a value one greater than the current highest
            block offset. The directory present field will always be True when
            the path present field is True. The directory present field does
            NOT indicate that the directory is present in the searched tree,
            rather it indicates that there are at least some files in some
            tree present there.
        """
        self._read_dirblocks_if_needed()
        key = (dirname, basename, b'')
        block_index, present = self._find_block_index_from_key(key)
        if not present:
            return (block_index, 0, False, False)
        block = self._dirblocks[block_index][1]
        entry_index, present = self._find_entry_index(key, block)
        while entry_index < len(block) and block[entry_index][0][1] == basename:
            if block[entry_index][1][tree_index][0] not in (b'a', b'r'):
                return (block_index, entry_index, True, True)
            entry_index += 1
        return (block_index, entry_index, True, False)

    def _get_entry(self, tree_index, fileid_utf8=None, path_utf8=None, include_deleted=False):
        """Get the dirstate entry for path in tree tree_index.

        If either file_id or path is supplied, it is used as the key to lookup.
        If both are supplied, the fastest lookup is used, and an error is
        raised if they do not both point at the same row.

        :param tree_index: The index of the tree we wish to locate this path
            in. If the path is present in that tree, the entry containing its
            details is returned, otherwise (None, None) is returned
            0 is the working tree, higher indexes are successive parent
            trees.
        :param fileid_utf8: A utf8 file_id to look up.
        :param path_utf8: An utf8 path to be looked up.
        :param include_deleted: If True, and performing a lookup via
            fileid_utf8 rather than path_utf8, return an entry for deleted
            (absent) paths.
        :return: The dirstate entry tuple for path, or (None, None)
        """
        self._read_dirblocks_if_needed()
        if path_utf8 is not None:
            if not isinstance(path_utf8, bytes):
                raise errors.BzrError('path_utf8 is not bytes: %s %r' % (type(path_utf8), path_utf8))
            dirname, basename = osutils.split(path_utf8)
            block_index, entry_index, dir_present, file_present = self._get_block_entry_index(dirname, basename, tree_index)
            if not file_present:
                return (None, None)
            entry = self._dirblocks[block_index][1][entry_index]
            if not (entry[0][2] and entry[1][tree_index][0] not in (b'a', b'r')):
                raise AssertionError('unversioned entry?')
            if fileid_utf8:
                if entry[0][2] != fileid_utf8:
                    self._changes_aborted = True
                    raise errors.BzrError('integrity error ? : mismatching tree_index, file_id and path')
            return entry
        else:
            possible_keys = self._get_id_index().get(fileid_utf8, ())
            if not possible_keys:
                return (None, None)
            for key in possible_keys:
                block_index, present = self._find_block_index_from_key(key)
                if not present:
                    continue
                block = self._dirblocks[block_index][1]
                entry_index, present = self._find_entry_index(key, block)
                if present:
                    entry = self._dirblocks[block_index][1][entry_index]
                    if entry[1][tree_index][0] in {b'f', b'd', b'l', b't'}:
                        return entry
                    if entry[1][tree_index][0] == b'a':
                        if include_deleted:
                            return entry
                        return (None, None)
                    if entry[1][tree_index][0] != b'r':
                        raise AssertionError('entry %r has invalid minikind %r for tree %r' % (entry, entry[1][tree_index][0], tree_index))
                    real_path = entry[1][tree_index][1]
                    return self._get_entry(tree_index, fileid_utf8=fileid_utf8, path_utf8=real_path)
            return (None, None)

    @classmethod
    def initialize(cls, path, sha1_provider=None):
        """Create a new dirstate on path.

        The new dirstate will be an empty tree - that is it has no parents,
        and only a root node - which has id ROOT_ID.

        :param path: The name of the file for the dirstate.
        :param sha1_provider: an object meeting the SHA1Provider interface.
            If None, a DefaultSHA1Provider is used.
        :return: A write-locked DirState object.
        """
        if sha1_provider is None:
            sha1_provider = DefaultSHA1Provider()
        result = cls(path, sha1_provider)
        empty_tree_dirblocks = [(b'', []), (b'', [])]
        empty_tree_dirblocks[0][1].append(((b'', b'', inventory.ROOT_ID), [(b'd', b'', 0, False, DirState.NULLSTAT)]))
        result.lock_write()
        try:
            result._set_data([], empty_tree_dirblocks)
            result.save()
        except:
            result.unlock()
            raise
        return result

    @staticmethod
    def _inv_entry_to_details(inv_entry):
        """Convert an inventory entry (from a revision tree) to state details.

        :param inv_entry: An inventory entry whose sha1 and link targets can be
            relied upon, and which has a revision set.
        :return: A details tuple - the details for a single tree at a path +
            id.
        """
        kind = inv_entry.kind
        minikind = DirState._kind_to_minikind[kind]
        tree_data = inv_entry.revision
        if kind == 'directory':
            fingerprint = b''
            size = 0
            executable = False
        elif kind == 'symlink':
            if inv_entry.symlink_target is None:
                fingerprint = b''
            else:
                fingerprint = inv_entry.symlink_target.encode('utf8')
            size = 0
            executable = False
        elif kind == 'file':
            fingerprint = inv_entry.text_sha1 or b''
            size = inv_entry.text_size or 0
            executable = inv_entry.executable
        elif kind == 'tree-reference':
            fingerprint = inv_entry.reference_revision or b''
            size = 0
            executable = False
        else:
            raise Exception("can't pack %s" % inv_entry)
        return static_tuple.StaticTuple(minikind, fingerprint, size, executable, tree_data)

    def _iter_child_entries(self, tree_index, path_utf8):
        """Iterate over all the entries that are children of path_utf.

        This only returns entries that are present (not in b'a', b'r') in
        tree_index. tree_index data is not refreshed, so if tree 0 is used,
        results may differ from that obtained if paths were statted to
        determine what ones were directories.

        Asking for the children of a non-directory will return an empty
        iterator.
        """
        pending_dirs = []
        next_pending_dirs = [path_utf8]
        absent = (b'a', b'r')
        while next_pending_dirs:
            pending_dirs = next_pending_dirs
            next_pending_dirs = []
            for path in pending_dirs:
                block_index, present = self._find_block_index_from_key((path, b'', b''))
                if block_index == 0:
                    block_index = 1
                    if len(self._dirblocks) == 1:
                        return
                if not present:
                    continue
                block = self._dirblocks[block_index]
                for entry in block[1]:
                    kind = entry[1][tree_index][0]
                    if kind not in absent:
                        yield entry
                    if kind == b'd':
                        if entry[0][0]:
                            path = entry[0][0] + b'/' + entry[0][1]
                        else:
                            path = entry[0][1]
                        next_pending_dirs.append(path)

    def _iter_entries(self):
        """Iterate over all the entries in the dirstate.

        Each yelt item is an entry in the standard format described in the
        docstring of breezy.dirstate.
        """
        self._read_dirblocks_if_needed()
        for directory in self._dirblocks:
            yield from directory[1]

    def _get_id_index(self):
        """Get an id index of self._dirblocks.

        This maps from file_id => [(directory, name, file_id)] entries where
        that file_id appears in one of the trees.
        """
        if self._id_index is None:
            id_index = {}
            for key, tree_details in self._iter_entries():
                self._add_to_id_index(id_index, key)
            self._id_index = id_index
        return self._id_index

    def _add_to_id_index(self, id_index, entry_key):
        """Add this entry to the _id_index mapping."""
        file_id = entry_key[2]
        entry_key = static_tuple.StaticTuple.from_sequence(entry_key)
        if file_id not in id_index:
            id_index[file_id] = static_tuple.StaticTuple(entry_key)
        else:
            entry_keys = id_index[file_id]
            if entry_key not in entry_keys:
                id_index[file_id] = entry_keys + (entry_key,)

    def _remove_from_id_index(self, id_index, entry_key):
        """Remove this entry from the _id_index mapping.

        It is an programming error to call this when the entry_key is not
        already present.
        """
        file_id = entry_key[2]
        entry_keys = list(id_index[file_id])
        entry_keys.remove(entry_key)
        id_index[file_id] = static_tuple.StaticTuple.from_sequence(entry_keys)

    def _get_output_lines(self, lines):
        """Format lines for final output.

        :param lines: A sequence of lines containing the parents list and the
            path lines.
        """
        output_lines = [DirState.HEADER_FORMAT_3]
        lines.append(b'')
        inventory_text = b'\x00\n\x00'.join(lines)
        output_lines.append(b'crc32: %d\n' % (zlib.crc32(inventory_text),))
        num_entries = len(lines) - 3
        output_lines.append(b'num_entries: %d\n' % (num_entries,))
        output_lines.append(inventory_text)
        return output_lines

    def _make_deleted_row(self, fileid_utf8, parents):
        """Return a deleted row for fileid_utf8."""
        return ((b'/', b'RECYCLED.BIN', b'file', fileid_utf8, 0, DirState.NULLSTAT, b''), parents)

    def _num_present_parents(self):
        """The number of parent entries in each record row."""
        return len(self._parents) - len(self._ghosts)

    @classmethod
    def on_file(cls, path, sha1_provider=None, worth_saving_limit=0, use_filesystem_for_exec=True):
        """Construct a DirState on the file at path "path".

        :param path: The path at which the dirstate file on disk should live.
        :param sha1_provider: an object meeting the SHA1Provider interface.
            If None, a DefaultSHA1Provider is used.
        :param worth_saving_limit: when the exact number of hash changed
            entries is known, only bother saving the dirstate if more than
            this count of entries have changed. -1 means never save.
        :param use_filesystem_for_exec: Whether to trust the filesystem
            for executable bit information
        :return: An unlocked DirState object, associated with the given path.
        """
        if sha1_provider is None:
            sha1_provider = DefaultSHA1Provider()
        result = cls(path, sha1_provider, worth_saving_limit=worth_saving_limit, use_filesystem_for_exec=use_filesystem_for_exec)
        return result

    def _read_dirblocks_if_needed(self):
        """Read in all the dirblocks from the file if they are not in memory.

        This populates self._dirblocks, and sets self._dirblock_state to
        IN_MEMORY_UNMODIFIED. It is not currently ready for incremental block
        loading.
        """
        self._read_header_if_needed()
        if self._dirblock_state == DirState.NOT_IN_MEMORY:
            _read_dirblocks(self)

    def _read_header(self):
        """This reads in the metadata header, and the parent ids.

        After reading in, the file should be positioned at the null
        just before the start of the first record in the file.

        :return: (expected crc checksum, number of entries, parent list)
        """
        self._read_prelude()
        parent_line = self._state_file.readline()
        info = parent_line.split(b'\x00')
        num_parents = int(info[0])
        self._parents = info[1:-1]
        ghost_line = self._state_file.readline()
        info = ghost_line.split(b'\x00')
        num_ghosts = int(info[1])
        self._ghosts = info[2:-1]
        self._header_state = DirState.IN_MEMORY_UNMODIFIED
        self._end_of_header = self._state_file.tell()

    def _read_header_if_needed(self):
        """Read the header of the dirstate file if needed."""
        if not self._lock_token:
            raise errors.ObjectNotLocked(self)
        if self._header_state == DirState.NOT_IN_MEMORY:
            self._read_header()

    def _read_prelude(self):
        """Read in the prelude header of the dirstate file.

        This only reads in the stuff that is not connected to the crc
        checksum. The position will be correct to read in the rest of
        the file and check the checksum after this point.
        The next entry in the file should be the number of parents,
        and their ids. Followed by a newline.
        """
        header = self._state_file.readline()
        if header != DirState.HEADER_FORMAT_3:
            raise errors.BzrError('invalid header line: {!r}'.format(header))
        crc_line = self._state_file.readline()
        if not crc_line.startswith(b'crc32: '):
            raise errors.BzrError('missing crc32 checksum: %r' % crc_line)
        self.crc_expected = int(crc_line[len(b'crc32: '):-1])
        num_entries_line = self._state_file.readline()
        if not num_entries_line.startswith(b'num_entries: '):
            raise errors.BzrError('missing num_entries line')
        self._num_entries = int(num_entries_line[len(b'num_entries: '):-1])

    def sha1_from_stat(self, path, stat_result):
        """Find a sha1 given a stat lookup."""
        return self._get_packed_stat_index().get(pack_stat(stat_result), None)

    def _get_packed_stat_index(self):
        """Get a packed_stat index of self._dirblocks."""
        if self._packed_stat_index is None:
            index = {}
            for key, tree_details in self._iter_entries():
                if tree_details[0][0] == b'f':
                    index[tree_details[0][4]] = tree_details[0][1]
            self._packed_stat_index = index
        return self._packed_stat_index

    def save(self):
        """Save any pending changes created during this session.

        We reuse the existing file, because that prevents race conditions with
        file creation, and use oslocks on it to prevent concurrent modification
        and reads - because dirstate's incremental data aggregation is not
        compatible with reading a modified file, and replacing a file in use by
        another process is impossible on Windows.

        A dirstate in read only mode should be smart enough though to validate
        that the file has not changed, and otherwise discard its cache and
        start over, to allow for fine grained read lock duration, so 'status'
        wont block 'commit' - for example.
        """
        if self._changes_aborted:
            trace.mutter('Not saving DirState because _changes_aborted is set.')
            return
        if not self._worth_saving():
            return
        grabbed_write_lock = False
        if self._lock_state != 'w':
            grabbed_write_lock, new_lock = self._lock_token.temporary_write_lock()
            self._lock_token = new_lock
            self._state_file = new_lock.f
            if not grabbed_write_lock:
                return
        try:
            lines = self.get_lines()
            self._state_file.seek(0)
            self._state_file.writelines(lines)
            self._state_file.truncate()
            self._state_file.flush()
            self._maybe_fdatasync()
            self._mark_unmodified()
        finally:
            if grabbed_write_lock:
                self._lock_token = self._lock_token.restore_read_lock()
                self._state_file = self._lock_token.f

    def _maybe_fdatasync(self):
        """Flush to disk if possible and if not configured off."""
        if self._config_stack.get('dirstate.fdatasync'):
            osutils.fdatasync(self._state_file.fileno())

    def _worth_saving(self):
        """Is it worth saving the dirstate or not?"""
        if self._header_state == DirState.IN_MEMORY_MODIFIED or self._dirblock_state == DirState.IN_MEMORY_MODIFIED:
            return True
        if self._dirblock_state == DirState.IN_MEMORY_HASH_MODIFIED:
            if self._worth_saving_limit == -1:
                return False
            if len(self._known_hash_changes) >= self._worth_saving_limit:
                return True
        return False

    def _set_data(self, parent_ids, dirblocks):
        """Set the full dirstate data in memory.

        This is an internal function used to completely replace the objects
        in memory state. It puts the dirstate into state 'full-dirty'.

        :param parent_ids: A list of parent tree revision ids.
        :param dirblocks: A list containing one tuple for each directory in the
            tree. Each tuple contains the directory path and a list of entries
            found in that directory.
        """
        self._dirblocks = dirblocks
        self._mark_modified(header_modified=True)
        self._parents = list(parent_ids)
        self._id_index = None
        self._packed_stat_index = None

    def set_path_id(self, path, new_id):
        """Change the id of path to new_id in the current working tree.

        :param path: The path inside the tree to set - b'' is the root, 'foo'
            is the path foo in the root.
        :param new_id: The new id to assign to the path. This must be a utf8
            file id (not unicode, and not None).
        """
        self._read_dirblocks_if_needed()
        if len(path):
            raise NotImplementedError(self.set_path_id)
        entry = self._get_entry(0, path_utf8=path)
        if entry[0][2] == new_id:
            return
        if new_id.__class__ != bytes:
            raise AssertionError('must be a utf8 file_id not {}'.format(type(new_id)))
        self._make_absent(entry)
        self.update_minimal((b'', b'', new_id), b'd', path_utf8=b'', packed_stat=entry[1][0][4])
        self._mark_modified()

    def set_parent_trees(self, trees, ghosts):
        """Set the parent trees for the dirstate.

        :param trees: A list of revision_id, tree tuples. tree must be provided
            even if the revision_id refers to a ghost: supply an empty tree in
            this case.
        :param ghosts: A list of the revision_ids that are ghosts at the time
            of setting.
        """
        self._read_dirblocks_if_needed()
        by_path = {}
        id_index = {}
        parent_trees = [tree for rev_id, tree in trees if rev_id not in ghosts]
        parent_count = len(parent_trees)
        st = static_tuple.StaticTuple
        for entry in self._iter_entries():
            if entry[1][0][0] in (b'a', b'r'):
                continue
            by_path[entry[0]] = [entry[1][0]] + [DirState.NULL_PARENT_DETAILS] * parent_count
            self._add_to_id_index(id_index, entry[0])
        for tree_index, tree in enumerate(parent_trees):
            tree_index = tree_index + 1
            new_location_suffix = [DirState.NULL_PARENT_DETAILS] * (parent_count - tree_index)
            last_dirname = None
            for path, entry in tree.iter_entries_by_dir():
                file_id = entry.file_id
                path_utf8 = path.encode('utf8')
                dirname, basename = osutils.split(path_utf8)
                if dirname == last_dirname:
                    dirname = last_dirname
                else:
                    last_dirname = dirname
                new_entry_key = st(dirname, basename, file_id)
                entry_keys = id_index.get(file_id, ())
                for entry_key in entry_keys:
                    if entry_key != new_entry_key:
                        by_path[entry_key][tree_index] = st(b'r', path_utf8, 0, False, b'')
                if new_entry_key in entry_keys:
                    by_path[new_entry_key][tree_index] = self._inv_entry_to_details(entry)
                else:
                    new_details = []
                    for lookup_index in range(tree_index):
                        if not len(entry_keys):
                            new_details.append(DirState.NULL_PARENT_DETAILS)
                        else:
                            a_key = next(iter(entry_keys))
                            if by_path[a_key][lookup_index][0] in (b'r', b'a'):
                                new_details.append(by_path[a_key][lookup_index])
                            else:
                                real_path = b'/'.join(a_key[0:2]).strip(b'/')
                                new_details.append(st(b'r', real_path, 0, False, b''))
                    new_details.append(self._inv_entry_to_details(entry))
                    new_details.extend(new_location_suffix)
                    by_path[new_entry_key] = new_details
                    self._add_to_id_index(id_index, new_entry_key)
        new_entries = self._sort_entries(by_path.items())
        self._entries_to_current_state(new_entries)
        self._parents = [rev_id for rev_id, tree in trees]
        self._ghosts = list(ghosts)
        self._mark_modified(header_modified=True)
        self._id_index = id_index

    def _sort_entries(self, entry_list):
        """Given a list of entries, sort them into the right order.

        This is done when constructing a new dirstate from trees - normally we
        try to keep everything in sorted blocks all the time, but sometimes
        it's easier to sort after the fact.
        """
        split_dirs = {}

        def _key(entry, _split_dirs=split_dirs, _st=static_tuple.StaticTuple):
            dirpath, fname, file_id = entry[0]
            try:
                split = _split_dirs[dirpath]
            except KeyError:
                split = _st.from_sequence(dirpath.split(b'/'))
                _split_dirs[dirpath] = split
            return _st(split, fname, file_id)
        return sorted(entry_list, key=_key)

    def set_state_from_inventory(self, new_inv):
        """Set new_inv as the current state.

        This API is called by tree transform, and will usually occur with
        existing parent trees.

        :param new_inv: The inventory object to set current state from.
        """
        if 'evil' in debug.debug_flags:
            trace.mutter_callsite(1, 'set_state_from_inventory called; please mutate the tree instead')
        tracing = 'dirstate' in debug.debug_flags
        if tracing:
            trace.mutter('set_state_from_inventory trace:')
        self._read_dirblocks_if_needed()
        new_iterator = new_inv.iter_entries_by_dir()
        old_iterator = iter(list(self._iter_entries()))
        current_new = next(new_iterator)
        current_old = next(old_iterator)

        def advance(iterator):
            try:
                return next(iterator)
            except StopIteration:
                return None
        while current_new or current_old:
            if current_old and current_old[1][0][0] in (b'a', b'r'):
                current_old = advance(old_iterator)
                continue
            if current_new:
                new_path_utf8 = current_new[0].encode('utf8')
                new_dirname, new_basename = osutils.split(new_path_utf8)
                new_id = current_new[1].file_id
                new_entry_key = (new_dirname, new_basename, new_id)
                current_new_minikind = DirState._kind_to_minikind[current_new[1].kind]
                if current_new_minikind == b't':
                    fingerprint = current_new[1].reference_revision or b''
                else:
                    fingerprint = b''
            else:
                new_path_utf8 = new_dirname = new_basename = new_id = new_entry_key = None
            if not current_old:
                if tracing:
                    trace.mutter("Appending from new '%s'.", new_path_utf8.decode('utf8'))
                self.update_minimal(new_entry_key, current_new_minikind, executable=current_new[1].executable, path_utf8=new_path_utf8, fingerprint=fingerprint, fullscan=True)
                current_new = advance(new_iterator)
            elif not current_new:
                if tracing:
                    trace.mutter("Truncating from old '%s/%s'.", current_old[0][0].decode('utf8'), current_old[0][1].decode('utf8'))
                self._make_absent(current_old)
                current_old = advance(old_iterator)
            elif new_entry_key == current_old[0]:
                if current_old[1][0][3] != current_new[1].executable or current_old[1][0][0] != current_new_minikind:
                    if tracing:
                        trace.mutter("Updating in-place change '%s'.", new_path_utf8.decode('utf8'))
                    self.update_minimal(current_old[0], current_new_minikind, executable=current_new[1].executable, path_utf8=new_path_utf8, fingerprint=fingerprint, fullscan=True)
                current_old = advance(old_iterator)
                current_new = advance(new_iterator)
            elif lt_by_dirs(new_dirname, current_old[0][0]) or (new_dirname == current_old[0][0] and new_entry_key[1:] < current_old[0][1:]):
                if tracing:
                    trace.mutter("Inserting from new '%s'.", new_path_utf8.decode('utf8'))
                self.update_minimal(new_entry_key, current_new_minikind, executable=current_new[1].executable, path_utf8=new_path_utf8, fingerprint=fingerprint, fullscan=True)
                current_new = advance(new_iterator)
            else:
                if tracing:
                    trace.mutter("Deleting from old '%s/%s'.", current_old[0][0].decode('utf8'), current_old[0][1].decode('utf8'))
                self._make_absent(current_old)
                current_old = advance(old_iterator)
        self._mark_modified()
        self._id_index = None
        self._packed_stat_index = None
        if tracing:
            trace.mutter('set_state_from_inventory complete.')

    def set_state_from_scratch(self, working_inv, parent_trees, parent_ghosts):
        """Wipe the currently stored state and set it to something new.

        This is a hard-reset for the data we are working with.
        """
        self._requires_lock()
        empty_root = ((b'', b'', inventory.ROOT_ID), [(b'd', b'', 0, False, DirState.NULLSTAT)])
        empty_tree_dirblocks = [(b'', [empty_root]), (b'', [])]
        self._set_data([], empty_tree_dirblocks)
        self.set_state_from_inventory(working_inv)
        self.set_parent_trees(parent_trees, parent_ghosts)

    def _make_absent(self, current_old):
        """Mark current_old - an entry - as absent for tree 0.

        :return: True if this was the last details entry for the entry key:
            that is, if the underlying block has had the entry removed, thus
            shrinking in length.
        """
        all_remaining_keys = set()
        for details in current_old[1][1:]:
            if details[0] not in (b'a', b'r'):
                all_remaining_keys.add(current_old[0])
            elif details[0] == b'r':
                all_remaining_keys.add(tuple(osutils.split(details[1])) + (current_old[0][2],))
        last_reference = current_old[0] not in all_remaining_keys
        if last_reference:
            block = self._find_block(current_old[0])
            entry_index, present = self._find_entry_index(current_old[0], block[1])
            if not present:
                raise AssertionError('could not find entry for {}'.format(current_old))
            block[1].pop(entry_index)
            if self._id_index is not None:
                self._remove_from_id_index(self._id_index, current_old[0])
        for update_key in all_remaining_keys:
            update_block_index, present = self._find_block_index_from_key(update_key)
            if not present:
                raise AssertionError('could not find block for {}'.format(update_key))
            update_entry_index, present = self._find_entry_index(update_key, self._dirblocks[update_block_index][1])
            if not present:
                raise AssertionError('could not find entry for {}'.format(update_key))
            update_tree_details = self._dirblocks[update_block_index][1][update_entry_index][1]
            if update_tree_details[0][0] == b'a':
                raise AssertionError('bad row {!r}'.format(update_tree_details))
            update_tree_details[0] = DirState.NULL_PARENT_DETAILS
        self._mark_modified()
        return last_reference

    def update_minimal(self, key, minikind, executable=False, fingerprint=b'', packed_stat=None, size=0, path_utf8=None, fullscan=False):
        """Update an entry to the state in tree 0.

        This will either create a new entry at 'key' or update an existing one.
        It also makes sure that any other records which might mention this are
        updated as well.

        :param key: (dir, name, file_id) for the new entry
        :param minikind: The type for the entry (b'f' == 'file', b'd' ==
                'directory'), etc.
        :param executable: Should the executable bit be set?
        :param fingerprint: Simple fingerprint for new entry: canonical-form
            sha1 for files, referenced revision id for subtrees, etc.
        :param packed_stat: Packed stat value for new entry.
        :param size: Size information for new entry
        :param path_utf8: key[0] + '/' + key[1], just passed in to avoid doing
                extra computation.
        :param fullscan: If True then a complete scan of the dirstate is being
            done and checking for duplicate rows should not be done. This
            should only be set by set_state_from_inventory and similar methods.

        If packed_stat and fingerprint are not given, they're invalidated in
        the entry.
        """
        block = self._find_block(key)[1]
        if packed_stat is None:
            packed_stat = DirState.NULLSTAT
        entry_index, present = self._find_entry_index(key, block)
        new_details = (minikind, fingerprint, size, executable, packed_stat)
        id_index = self._get_id_index()
        if not present:
            if not fullscan:
                low_index, _ = self._find_entry_index(key[0:2] + (b'',), block)
                while low_index < len(block):
                    entry = block[low_index]
                    if entry[0][0:2] == key[0:2]:
                        if entry[1][0][0] not in (b'a', b'r'):
                            self._raise_invalid((b'%s/%s' % key[0:2]).decode('utf8'), key[2], 'Attempt to add item at path already occupied by id %r' % entry[0][2])
                        low_index += 1
                    else:
                        break
            existing_keys = id_index.get(key[2], ())
            if not existing_keys:
                new_entry = (key, [new_details] + self._empty_parent_info())
            else:
                new_entry = (key, [new_details])
                for other_key in tuple(existing_keys):
                    other_block_index, present = self._find_block_index_from_key(other_key)
                    if not present:
                        raise AssertionError('could not find block for {}'.format(other_key))
                    other_block = self._dirblocks[other_block_index][1]
                    other_entry_index, present = self._find_entry_index(other_key, other_block)
                    if not present:
                        raise AssertionError('update_minimal: could not find other entry for %s' % (other_key,))
                    if path_utf8 is None:
                        raise AssertionError('no path')
                    other_entry = other_block[other_entry_index]
                    other_entry[1][0] = (b'r', path_utf8, 0, False, b'')
                    if self._maybe_remove_row(other_block, other_entry_index, id_index):
                        entry_index, _ = self._find_entry_index(key, block)
                num_present_parents = self._num_present_parents()
                if num_present_parents:
                    other_key = list(existing_keys)[0]
                for lookup_index in range(1, num_present_parents + 1):
                    update_block_index, present = self._find_block_index_from_key(other_key)
                    if not present:
                        raise AssertionError('could not find block for {}'.format(other_key))
                    update_entry_index, present = self._find_entry_index(other_key, self._dirblocks[update_block_index][1])
                    if not present:
                        raise AssertionError('update_minimal: could not find entry for {}'.format(other_key))
                    update_details = self._dirblocks[update_block_index][1][update_entry_index][1][lookup_index]
                    if update_details[0] in (b'a', b'r'):
                        new_entry[1].append(update_details)
                    else:
                        pointer_path = osutils.pathjoin(*other_key[0:2])
                        new_entry[1].append((b'r', pointer_path, 0, False, b''))
            block.insert(entry_index, new_entry)
            self._add_to_id_index(id_index, key)
        else:
            block[entry_index][1][0] = new_details
            if path_utf8 is None:
                raise AssertionError('no path')
            existing_keys = id_index.get(key[2], ())
            if key not in existing_keys:
                raise AssertionError('We found the entry in the blocks, but the key is not in the id_index. key: %s, existing_keys: %s' % (key, existing_keys))
            for entry_key in existing_keys:
                if entry_key != key:
                    block_index, present = self._find_block_index_from_key(entry_key)
                    if not present:
                        raise AssertionError('not present: %r', entry_key)
                    entry_index, present = self._find_entry_index(entry_key, self._dirblocks[block_index][1])
                    if not present:
                        raise AssertionError('not present: %r', entry_key)
                    self._dirblocks[block_index][1][entry_index][1][0] = (b'r', path_utf8, 0, False, b'')
        if new_details[0] == b'd':
            subdir_key = (osutils.pathjoin(*key[0:2]), b'', b'')
            block_index, present = self._find_block_index_from_key(subdir_key)
            if not present:
                self._dirblocks.insert(block_index, (subdir_key[0], []))
        self._mark_modified()

    def _maybe_remove_row(self, block, index, id_index):
        """Remove index if it is absent or relocated across the row.

        id_index is updated accordingly.
        :return: True if we removed the row, False otherwise
        """
        present_in_row = False
        entry = block[index]
        for column in entry[1]:
            if column[0] not in (b'a', b'r'):
                present_in_row = True
                break
        if not present_in_row:
            block.pop(index)
            self._remove_from_id_index(id_index, entry[0])
            return True
        return False

    def _validate(self):
        """Check that invariants on the dirblock are correct.

        This can be useful in debugging; it shouldn't be necessary in
        normal code.

        This must be called with a lock held.
        """
        from pprint import pformat
        self._read_dirblocks_if_needed()
        if len(self._dirblocks) > 0:
            if not self._dirblocks[0][0] == b'':
                raise AssertionError("dirblocks don't start with root block:\n" + pformat(self._dirblocks))
        if len(self._dirblocks) > 1:
            if not self._dirblocks[1][0] == b'':
                raise AssertionError('dirblocks missing root directory:\n' + pformat(self._dirblocks))
        dir_names = [d[0].split(b'/') for d in self._dirblocks[1:]]
        if dir_names != sorted(dir_names):
            raise AssertionError('dir names are not in sorted order:\n' + pformat(self._dirblocks) + '\nkeys:\n' + pformat(dir_names))
        for dirblock in self._dirblocks:
            for entry in dirblock[1]:
                if dirblock[0] != entry[0][0]:
                    raise AssertionError("entry key for %rdoesn't match directory name in\n%r" % (entry, pformat(dirblock)))
            if dirblock[1] != sorted(dirblock[1]):
                raise AssertionError('dirblock for %r is not sorted:\n%s' % (dirblock[0], pformat(dirblock)))

        def check_valid_parent():
            """Check that the current entry has a valid parent.

            This makes sure that the parent has a record,
            and that the parent isn't marked as "absent" in the
            current tree. (It is invalid to have a non-absent file in an absent
            directory.)
            """
            if entry[0][0:2] == (b'', b''):
                return
            parent_entry = self._get_entry(tree_index, path_utf8=entry[0][0])
            if parent_entry == (None, None):
                raise AssertionError('no parent entry for: %s in tree %s' % (this_path, tree_index))
            if parent_entry[1][tree_index][0] != b'd':
                raise AssertionError('Parent entry for %s is not marked as a valid directory. %s' % (this_path, parent_entry))
        tree_count = self._num_present_parents() + 1
        id_path_maps = [{} for _ in range(tree_count)]
        for entry in self._iter_entries():
            file_id = entry[0][2]
            this_path = osutils.pathjoin(entry[0][0], entry[0][1])
            if len(entry[1]) != tree_count:
                raise AssertionError('wrong number of entry details for row\n%s,\nexpected %d' % (pformat(entry), tree_count))
            absent_positions = 0
            for tree_index, tree_state in enumerate(entry[1]):
                this_tree_map = id_path_maps[tree_index]
                minikind = tree_state[0]
                if minikind in (b'a', b'r'):
                    absent_positions += 1
                if file_id in this_tree_map:
                    previous_path, previous_loc = this_tree_map[file_id]
                    if minikind == b'a':
                        if previous_path is not None:
                            raise AssertionError('file %s is absent in row %r but also present at %r' % (file_id.decode('utf-8'), entry, previous_path))
                    elif minikind == b'r':
                        target_location = tree_state[1]
                        if previous_path != target_location:
                            raise AssertionError('file %s relocation in row %r but also at %r' % (file_id, entry, previous_path))
                    else:
                        if previous_path != this_path:
                            raise AssertionError('entry %r inconsistent with previous path %r seen at %r' % (entry, previous_path, previous_loc))
                        check_valid_parent()
                elif minikind == b'a':
                    this_tree_map[file_id] = (None, this_path)
                elif minikind == b'r':
                    this_tree_map[file_id] = (tree_state[1], this_path)
                else:
                    this_tree_map[file_id] = (this_path, this_path)
                    check_valid_parent()
            if absent_positions == tree_count:
                raise AssertionError('entry {!r} has no data for any tree.'.format(entry))
        if self._id_index is not None:
            for file_id, entry_keys in self._id_index.items():
                for entry_key in entry_keys:
                    if entry_key[2] != file_id:
                        raise AssertionError('file_id %r did not match entry key %s' % (file_id, entry_key))
                    block_index, present = self._find_block_index_from_key(entry_key)
                    if not present:
                        raise AssertionError('missing block for entry key: %r', entry_key)
                    entry_index, present = self._find_entry_index(entry_key, self._dirblocks[block_index][1])
                    if not present:
                        raise AssertionError('missing entry for key: %r', entry_key)
                if len(entry_keys) != len(set(entry_keys)):
                    raise AssertionError('id_index contained non-unique data for %s' % (entry_keys,))

    def _wipe_state(self):
        """Forget all state information about the dirstate."""
        self._header_state = DirState.NOT_IN_MEMORY
        self._dirblock_state = DirState.NOT_IN_MEMORY
        self._changes_aborted = False
        self._parents = []
        self._ghosts = []
        self._dirblocks = []
        self._id_index = None
        self._packed_stat_index = None
        self._end_of_header = None
        self._cutoff_time = None
        self._split_path_cache = {}

    def lock_read(self):
        """Acquire a read lock on the dirstate."""
        if self._lock_token is not None:
            raise errors.LockContention(self._lock_token)
        self._lock_token = lock.ReadLock(self._filename)
        self._lock_state = 'r'
        self._state_file = self._lock_token.f
        self._wipe_state()
        return lock.LogicalLockResult(self.unlock)

    def lock_write(self):
        """Acquire a write lock on the dirstate."""
        if self._lock_token is not None:
            raise errors.LockContention(self._lock_token)
        self._lock_token = lock.WriteLock(self._filename)
        self._lock_state = 'w'
        self._state_file = self._lock_token.f
        self._wipe_state()
        return lock.LogicalLockResult(self.unlock, self._lock_token)

    def unlock(self):
        """Drop any locks held on the dirstate."""
        if self._lock_token is None:
            raise errors.LockNotHeld(self)
        self._state_file = None
        self._lock_state = None
        self._lock_token.unlock()
        self._lock_token = None
        self._split_path_cache = {}

    def _requires_lock(self):
        """Check that a lock is currently held by someone on the dirstate."""
        if not self._lock_token:
            raise errors.ObjectNotLocked(self)