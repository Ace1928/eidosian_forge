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
class ProcessEntryPython:
    __slots__ = ['old_dirname_to_file_id', 'new_dirname_to_file_id', 'last_source_parent', 'last_target_parent', 'include_unchanged', 'partial', 'use_filesystem_for_exec', 'utf8_decode', 'searched_specific_files', 'search_specific_files', 'searched_exact_paths', 'search_specific_file_parents', 'seen_ids', 'state', 'source_index', 'target_index', 'want_unversioned', 'tree']

    def __init__(self, include_unchanged, use_filesystem_for_exec, search_specific_files, state, source_index, target_index, want_unversioned, tree):
        self.old_dirname_to_file_id = {}
        self.new_dirname_to_file_id = {}
        self.partial = search_specific_files != {''}
        self.last_source_parent = [None, None]
        self.last_target_parent = [None, None]
        self.include_unchanged = include_unchanged
        self.use_filesystem_for_exec = use_filesystem_for_exec
        self.utf8_decode = codecs.utf_8_decode
        self.searched_specific_files = set()
        self.searched_exact_paths = set()
        self.search_specific_files = search_specific_files
        self.search_specific_file_parents = set()
        self.seen_ids = set()
        self.state = state
        self.source_index = source_index
        self.target_index = target_index
        if target_index != 0:
            raise errors.BzrError('unsupported target index')
        self.want_unversioned = want_unversioned
        self.tree = tree

    def _process_entry(self, entry, path_info, pathjoin=osutils.pathjoin):
        """Compare an entry and real disk to generate delta information.

        :param path_info: top_relpath, basename, kind, lstat, abspath for
            the path of entry. If None, then the path is considered absent in
            the target (Perhaps we should pass in a concrete entry for this ?)
            Basename is returned as a utf8 string because we expect this
            tuple will be ignored, and don't want to take the time to
            decode.
        :return: (iter_changes_result, changed). If the entry has not been
            handled then changed is None. Otherwise it is False if no content
            or metadata changes have occurred, and True if any content or
            metadata change has occurred. If self.include_unchanged is True then
            if changed is not None, iter_changes_result will always be a result
            tuple. Otherwise, iter_changes_result is None unless changed is
            True.
        """
        if self.source_index is None:
            source_details = DirState.NULL_PARENT_DETAILS
        else:
            source_details = entry[1][self.source_index]
        _fdltr = {b'f', b'd', b'l', b't', b'r'}
        _fdlt = {b'f', b'd', b'l', b't'}
        _ra = (b'r', b'a')
        target_details = entry[1][self.target_index]
        target_minikind = target_details[0]
        if path_info is not None and target_minikind in _fdlt:
            if not self.target_index == 0:
                raise AssertionError()
            link_or_sha1 = update_entry(self.state, entry, abspath=path_info[4], stat_value=path_info[3])
            target_details = entry[1][self.target_index]
            target_minikind = target_details[0]
        else:
            link_or_sha1 = None
        file_id = entry[0][2]
        source_minikind = source_details[0]
        if source_minikind in _fdltr and target_minikind in _fdlt:
            if source_minikind == b'r':
                if not osutils.is_inside_any(self.searched_specific_files, source_details[1]):
                    self.search_specific_files.add(source_details[1])
                old_path = source_details[1]
                old_dirname, old_basename = os.path.split(old_path)
                path = pathjoin(entry[0][0], entry[0][1])
                old_entry = self.state._get_entry(self.source_index, path_utf8=old_path)
                if old_entry == (None, None):
                    raise DirstateCorrupt(self.state._filename, "entry '%s/%s' is considered renamed from %r but source does not exist\nentry: %s" % (entry[0][0], entry[0][1], old_path, entry))
                source_details = old_entry[1][self.source_index]
                source_minikind = source_details[0]
            else:
                old_dirname = entry[0][0]
                old_basename = entry[0][1]
                old_path = path = None
            if path_info is None:
                content_change = True
                target_kind = None
                target_exec = False
            else:
                target_kind = path_info[2]
                if target_kind == 'directory':
                    if path is None:
                        old_path = path = pathjoin(old_dirname, old_basename)
                    self.new_dirname_to_file_id[path] = file_id
                    if source_minikind != b'd':
                        content_change = True
                    else:
                        content_change = False
                    target_exec = False
                elif target_kind == 'file':
                    if source_minikind != b'f':
                        content_change = True
                    else:
                        if link_or_sha1 is None:
                            statvalue, link_or_sha1 = self.state._sha1_provider.stat_and_sha1(path_info[4])
                            self.state._observed_sha1(entry, link_or_sha1, statvalue)
                        content_change = link_or_sha1 != source_details[1]
                    if self.use_filesystem_for_exec:
                        target_exec = bool(stat.S_IEXEC & path_info[3].st_mode)
                    else:
                        target_exec = target_details[3]
                elif target_kind == 'symlink':
                    if source_minikind != b'l':
                        content_change = True
                    else:
                        content_change = link_or_sha1 != source_details[1]
                    target_exec = False
                elif target_kind == 'tree-reference':
                    if source_minikind != b't':
                        content_change = True
                    else:
                        content_change = False
                    target_exec = False
                else:
                    if path is None:
                        path = pathjoin(old_dirname, old_basename)
                    raise errors.BadFileKindError(path, path_info[2])
            if source_minikind == b'd':
                if path is None:
                    old_path = path = pathjoin(old_dirname, old_basename)
                self.old_dirname_to_file_id[old_path] = file_id
            if old_basename and old_dirname == self.last_source_parent[0]:
                source_parent_id = self.last_source_parent[1]
            else:
                try:
                    source_parent_id = self.old_dirname_to_file_id[old_dirname]
                except KeyError:
                    source_parent_entry = self.state._get_entry(self.source_index, path_utf8=old_dirname)
                    source_parent_id = source_parent_entry[0][2]
                if source_parent_id == entry[0][2]:
                    source_parent_id = None
                else:
                    self.last_source_parent[0] = old_dirname
                    self.last_source_parent[1] = source_parent_id
            new_dirname = entry[0][0]
            if entry[0][1] and new_dirname == self.last_target_parent[0]:
                target_parent_id = self.last_target_parent[1]
            else:
                try:
                    target_parent_id = self.new_dirname_to_file_id[new_dirname]
                except KeyError:
                    target_parent_entry = self.state._get_entry(self.target_index, path_utf8=new_dirname)
                    if target_parent_entry == (None, None):
                        raise AssertionError('Could not find target parent in wt: %s\nparent of: %s' % (new_dirname, entry))
                    target_parent_id = target_parent_entry[0][2]
                if target_parent_id == entry[0][2]:
                    target_parent_id = None
                else:
                    self.last_target_parent[0] = new_dirname
                    self.last_target_parent[1] = target_parent_id
            source_exec = source_details[3]
            changed = content_change or source_parent_id != target_parent_id or old_basename != entry[0][1] or (source_exec != target_exec)
            if not changed and (not self.include_unchanged):
                return (None, False)
            else:
                if old_path is None:
                    old_path = path = pathjoin(old_dirname, old_basename)
                    old_path_u = self.utf8_decode(old_path, 'surrogateescape')[0]
                    path_u = old_path_u
                else:
                    old_path_u = self.utf8_decode(old_path, 'surrogateescape')[0]
                    if old_path == path:
                        path_u = old_path_u
                    else:
                        path_u = self.utf8_decode(path, 'surrogateescape')[0]
                source_kind = DirState._minikind_to_kind[source_minikind]
                return (InventoryTreeChange(entry[0][2], (old_path_u, path_u), content_change, (True, True), (source_parent_id, target_parent_id), (self.utf8_decode(old_basename, 'surrogateescape')[0], self.utf8_decode(entry[0][1], 'surrogateescape')[0]), (source_kind, target_kind), (source_exec, target_exec)), changed)
        elif source_minikind in b'a' and target_minikind in _fdlt:
            path = pathjoin(entry[0][0], entry[0][1])
            parent_id = self.state._get_entry(self.target_index, path_utf8=entry[0][0])[0][2]
            if parent_id == entry[0][2]:
                parent_id = None
            if path_info is not None:
                if self.use_filesystem_for_exec:
                    target_exec = bool(stat.S_ISREG(path_info[3].st_mode) and stat.S_IEXEC & path_info[3].st_mode)
                else:
                    target_exec = target_details[3]
                return (InventoryTreeChange(entry[0][2], (None, self.utf8_decode(path, 'surrogateescape')[0]), True, (False, True), (None, parent_id), (None, self.utf8_decode(entry[0][1], 'surrogateescape')[0]), (None, path_info[2]), (None, target_exec)), True)
            else:
                return (InventoryTreeChange(entry[0][2], (None, self.utf8_decode(path, 'surrogateescape')[0]), False, (False, True), (None, parent_id), (None, self.utf8_decode(entry[0][1], 'surrogateescape')[0]), (None, None), (None, False)), True)
        elif source_minikind in _fdlt and target_minikind in b'a':
            old_path = pathjoin(entry[0][0], entry[0][1])
            parent_id = self.state._get_entry(self.source_index, path_utf8=entry[0][0])[0][2]
            if parent_id == entry[0][2]:
                parent_id = None
            return (InventoryTreeChange(entry[0][2], (self.utf8_decode(old_path, 'surrogateescape')[0], None), True, (True, False), (parent_id, None), (self.utf8_decode(entry[0][1], 'surrogateescape')[0], None), (DirState._minikind_to_kind[source_minikind], None), (source_details[3], None)), True)
        elif source_minikind in _fdlt and target_minikind in b'r':
            if not osutils.is_inside_any(self.searched_specific_files, target_details[1]):
                self.search_specific_files.add(target_details[1])
        elif source_minikind in _ra and target_minikind in _ra:
            pass
        else:
            raise AssertionError("don't know how to compare source_minikind=%r, target_minikind=%r" % (source_minikind, target_minikind))
        return (None, None)

    def __iter__(self):
        return self

    def _gather_result_for_consistency(self, result):
        """Check a result we will yield to make sure we are consistent later.

        This gathers result's parents into a set to output later.

        :param result: A result tuple.
        """
        if not self.partial or not result.file_id:
            return
        self.seen_ids.add(result.file_id)
        new_path = result.path[1]
        if new_path:
            self.search_specific_file_parents.update((p.encode('utf8', 'surrogateescape') for p in osutils.parent_directories(new_path)))
            self.search_specific_file_parents.add(b'')

    def iter_changes(self):
        """Iterate over the changes."""
        utf8_decode = codecs.utf_8_decode
        _lt_by_dirs = lt_by_dirs
        _process_entry = self._process_entry
        search_specific_files = self.search_specific_files
        searched_specific_files = self.searched_specific_files
        splitpath = osutils.splitpath
        while search_specific_files:
            current_root = search_specific_files.pop()
            current_root_unicode = current_root.decode('utf8')
            searched_specific_files.add(current_root)
            root_entries = self.state._entries_for_path(current_root)
            root_abspath = self.tree.abspath(current_root_unicode)
            try:
                root_stat = os.lstat(root_abspath)
            except OSError as e:
                if e.errno == errno.ENOENT:
                    root_dir_info = None
                else:
                    raise
            else:
                root_dir_info = (b'', current_root, osutils.file_kind_from_stat_mode(root_stat.st_mode), root_stat, root_abspath)
                if root_dir_info[2] == 'directory':
                    if self.tree._directory_is_tree_reference(current_root.decode('utf8')):
                        root_dir_info = root_dir_info[:2] + ('tree-reference',) + root_dir_info[3:]
            if not root_entries and (not root_dir_info):
                continue
            path_handled = False
            for entry in root_entries:
                result, changed = _process_entry(entry, root_dir_info)
                if changed is not None:
                    path_handled = True
                    if changed:
                        self._gather_result_for_consistency(result)
                    if changed or self.include_unchanged:
                        yield result
            if self.want_unversioned and (not path_handled) and root_dir_info:
                new_executable = bool(stat.S_ISREG(root_dir_info[3].st_mode) and stat.S_IEXEC & root_dir_info[3].st_mode)
                yield InventoryTreeChange(None, (None, current_root_unicode), True, (False, False), (None, None), (None, splitpath(current_root_unicode)[-1]), (None, root_dir_info[2]), (None, new_executable))
            initial_key = (current_root, b'', b'')
            block_index, _ = self.state._find_block_index_from_key(initial_key)
            if block_index == 0:
                block_index += 1
            if root_dir_info and root_dir_info[2] == 'tree-reference':
                current_dir_info = None
            else:
                dir_iterator = osutils._walkdirs_utf8(root_abspath, prefix=current_root)
                try:
                    current_dir_info = next(dir_iterator)
                except OSError as e:
                    e_winerror = getattr(e, 'winerror', None)
                    win_errors = (ERROR_DIRECTORY, ERROR_PATH_NOT_FOUND)
                    if e.errno in (errno.ENOENT, errno.ENOTDIR, errno.EINVAL):
                        current_dir_info = None
                    elif sys.platform == 'win32' and (e.errno in win_errors or e_winerror in win_errors):
                        current_dir_info = None
                    else:
                        raise
                else:
                    if current_dir_info[0][0] == b'':
                        bzr_index = bisect.bisect_left(current_dir_info[1], (b'.bzr',))
                        if current_dir_info[1][bzr_index][0] != b'.bzr':
                            raise AssertionError()
                        del current_dir_info[1][bzr_index]
            if block_index < len(self.state._dirblocks) and osutils.is_inside(current_root, self.state._dirblocks[block_index][0]):
                current_block = self.state._dirblocks[block_index]
            else:
                current_block = None
            while current_dir_info is not None or current_block is not None:
                if current_dir_info and current_block and (current_dir_info[0][0] != current_block[0]):
                    if _lt_by_dirs(current_dir_info[0][0], current_block[0]):
                        path_index = 0
                        while path_index < len(current_dir_info[1]):
                            current_path_info = current_dir_info[1][path_index]
                            if self.want_unversioned:
                                if current_path_info[2] == 'directory':
                                    if self.tree._directory_is_tree_reference(current_path_info[0].decode('utf8')):
                                        current_path_info = current_path_info[:2] + ('tree-reference',) + current_path_info[3:]
                                new_executable = bool(stat.S_ISREG(current_path_info[3].st_mode) and stat.S_IEXEC & current_path_info[3].st_mode)
                                yield InventoryTreeChange(None, (None, utf8_decode(current_path_info[0], 'surrogateescape')[0]), True, (False, False), (None, None), (None, utf8_decode(current_path_info[1], 'surrogateescape')[0]), (None, current_path_info[2]), (None, new_executable))
                            if current_path_info[2] in ('directory', 'tree-reference'):
                                del current_dir_info[1][path_index]
                                path_index -= 1
                            path_index += 1
                        try:
                            current_dir_info = next(dir_iterator)
                        except StopIteration:
                            current_dir_info = None
                    else:
                        for current_entry in current_block[1]:
                            result, changed = _process_entry(current_entry, None)
                            if changed is not None:
                                if changed:
                                    self._gather_result_for_consistency(result)
                                if changed or self.include_unchanged:
                                    yield result
                        block_index += 1
                        if block_index < len(self.state._dirblocks) and osutils.is_inside(current_root, self.state._dirblocks[block_index][0]):
                            current_block = self.state._dirblocks[block_index]
                        else:
                            current_block = None
                    continue
                entry_index = 0
                if current_block and entry_index < len(current_block[1]):
                    current_entry = current_block[1][entry_index]
                else:
                    current_entry = None
                advance_entry = True
                path_index = 0
                if current_dir_info and path_index < len(current_dir_info[1]):
                    current_path_info = current_dir_info[1][path_index]
                    if current_path_info[2] == 'directory':
                        if self.tree._directory_is_tree_reference(current_path_info[0].decode('utf8')):
                            current_path_info = current_path_info[:2] + ('tree-reference',) + current_path_info[3:]
                else:
                    current_path_info = None
                advance_path = True
                path_handled = False
                while current_entry is not None or current_path_info is not None:
                    if current_entry is None:
                        pass
                    elif current_path_info is None:
                        result, changed = _process_entry(current_entry, current_path_info)
                        if changed is not None:
                            if changed:
                                self._gather_result_for_consistency(result)
                            if changed or self.include_unchanged:
                                yield result
                    elif current_entry[0][1] != current_path_info[1] or current_entry[1][self.target_index][0] in (b'a', b'r'):
                        if current_path_info[1] < current_entry[0][1]:
                            advance_entry = False
                        else:
                            result, changed = _process_entry(current_entry, None)
                            if changed is not None:
                                if changed:
                                    self._gather_result_for_consistency(result)
                                if changed or self.include_unchanged:
                                    yield result
                            advance_path = False
                    else:
                        result, changed = _process_entry(current_entry, current_path_info)
                        if changed is not None:
                            path_handled = True
                            if changed:
                                self._gather_result_for_consistency(result)
                            if changed or self.include_unchanged:
                                yield result
                    if advance_entry and current_entry is not None:
                        entry_index += 1
                        if entry_index < len(current_block[1]):
                            current_entry = current_block[1][entry_index]
                        else:
                            current_entry = None
                    else:
                        advance_entry = True
                    if advance_path and current_path_info is not None:
                        if not path_handled:
                            if self.want_unversioned:
                                new_executable = bool(stat.S_ISREG(current_path_info[3].st_mode) and stat.S_IEXEC & current_path_info[3].st_mode)
                                relpath_unicode = utf8_decode(current_path_info[0], 'surrogateescape')[0]
                                yield InventoryTreeChange(None, (None, relpath_unicode), True, (False, False), (None, None), (None, utf8_decode(current_path_info[1], 'surrogateescape')[0]), (None, current_path_info[2]), (None, new_executable))
                            if current_path_info[2] in 'directory':
                                del current_dir_info[1][path_index]
                                path_index -= 1
                        if current_path_info[2] == 'tree-reference':
                            del current_dir_info[1][path_index]
                            path_index -= 1
                        path_index += 1
                        if path_index < len(current_dir_info[1]):
                            current_path_info = current_dir_info[1][path_index]
                            if current_path_info[2] == 'directory':
                                if self.tree._directory_is_tree_reference(current_path_info[0].decode('utf8')):
                                    current_path_info = current_path_info[:2] + ('tree-reference',) + current_path_info[3:]
                        else:
                            current_path_info = None
                        path_handled = False
                    else:
                        advance_path = True
                if current_block is not None:
                    block_index += 1
                    if block_index < len(self.state._dirblocks) and osutils.is_inside(current_root, self.state._dirblocks[block_index][0]):
                        current_block = self.state._dirblocks[block_index]
                    else:
                        current_block = None
                if current_dir_info is not None:
                    try:
                        current_dir_info = next(dir_iterator)
                    except StopIteration:
                        current_dir_info = None
        for result in self._iter_specific_file_parents():
            yield result

    def _iter_specific_file_parents(self):
        """Iter over the specific file parents."""
        while self.search_specific_file_parents:
            path_utf8 = self.search_specific_file_parents.pop()
            if osutils.is_inside_any(self.searched_specific_files, path_utf8):
                continue
            if path_utf8 in self.searched_exact_paths:
                continue
            path_entries = self.state._entries_for_path(path_utf8)
            selected_entries = []
            found_item = False
            for candidate_entry in path_entries:
                if candidate_entry[1][self.target_index][0] not in (b'a', b'r'):
                    found_item = True
                    selected_entries.append(candidate_entry)
                elif self.source_index is not None and candidate_entry[1][self.source_index][0] not in (b'a', b'r'):
                    found_item = True
                    if candidate_entry[1][self.target_index][0] == b'a':
                        selected_entries.append(candidate_entry)
                    else:
                        self.search_specific_file_parents.add(candidate_entry[1][self.target_index][1])
            if not found_item:
                raise AssertionError('Missing entry for specific path parent {!r}, {!r}'.format(path_utf8, path_entries))
            path_info = self._path_info(path_utf8, path_utf8.decode('utf8'))
            for entry in selected_entries:
                if entry[0][2] in self.seen_ids:
                    continue
                result, changed = self._process_entry(entry, path_info)
                if changed is None:
                    raise AssertionError('Got entry<->path mismatch for specific path %r entry %r path_info %r ' % (path_utf8, entry, path_info))
                if changed:
                    self._gather_result_for_consistency(result)
                    if result.kind[0] == 'directory' and result.kind[1] != 'directory':
                        if entry[1][self.source_index][0] == b'r':
                            entry_path_utf8 = entry[1][self.source_index][1]
                        else:
                            entry_path_utf8 = path_utf8
                        initial_key = (entry_path_utf8, b'', b'')
                        block_index, _ = self.state._find_block_index_from_key(initial_key)
                        if block_index == 0:
                            block_index += 1
                        current_block = None
                        if block_index < len(self.state._dirblocks):
                            current_block = self.state._dirblocks[block_index]
                            if not osutils.is_inside(entry_path_utf8, current_block[0]):
                                current_block = None
                        if current_block is not None:
                            for entry in current_block[1]:
                                if entry[1][self.source_index][0] in (b'a', b'r'):
                                    continue
                                self.search_specific_file_parents.add(osutils.pathjoin(*entry[0][:2]))
                if changed or self.include_unchanged:
                    yield result
            self.searched_exact_paths.add(path_utf8)

    def _path_info(self, utf8_path, unicode_path):
        """Generate path_info for unicode_path.

        :return: None if unicode_path does not exist, or a path_info tuple.
        """
        abspath = self.tree.abspath(unicode_path)
        try:
            stat = os.lstat(abspath)
        except OSError as e:
            if e.errno == errno.ENOENT:
                return None
            else:
                raise
        utf8_basename = utf8_path.rsplit(b'/', 1)[-1]
        dir_info = (utf8_path, utf8_basename, osutils.file_kind_from_stat_mode(stat.st_mode), stat, abspath)
        if dir_info[2] == 'directory':
            if self.tree._directory_is_tree_reference(unicode_path):
                self.root_dir_info = self.root_dir_info[:2] + ('tree-reference',) + self.root_dir_info[3:]
        return dir_info