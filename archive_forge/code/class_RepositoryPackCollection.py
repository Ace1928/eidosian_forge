import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
class RepositoryPackCollection:
    """Management of packs within a repository.

    :ivar _names: map of {pack_name: (index_size,)}
    """
    pack_factory: Type[NewPack]
    resumed_pack_factory: Type[ResumedPack]
    normal_packer_class: Type[Packer]
    optimising_packer_class: Type[Packer]

    def __init__(self, repo, transport, index_transport, upload_transport, pack_transport, index_builder_class, index_class, use_chk_index):
        """Create a new RepositoryPackCollection.

        :param transport: Addresses the repository base directory
            (typically .bzr/repository/).
        :param index_transport: Addresses the directory containing indices.
        :param upload_transport: Addresses the directory into which packs are written
            while they're being created.
        :param pack_transport: Addresses the directory of existing complete packs.
        :param index_builder_class: The index builder class to use.
        :param index_class: The index class to use.
        :param use_chk_index: Whether to setup and manage a CHK index.
        """
        self.repo = repo
        self.transport = transport
        self._index_transport = index_transport
        self._upload_transport = upload_transport
        self._pack_transport = pack_transport
        self._index_builder_class = index_builder_class
        self._index_class = index_class
        self._suffix_offsets = {'.rix': 0, '.iix': 1, '.tix': 2, '.six': 3, '.cix': 4}
        self.packs = []
        self._names = None
        self._packs_by_name = {}
        self._packs_at_load = None
        self._new_pack = None
        flush = self._flush_new_pack
        self.revision_index = AggregateIndex(self.reload_pack_names, flush)
        self.inventory_index = AggregateIndex(self.reload_pack_names, flush)
        self.text_index = AggregateIndex(self.reload_pack_names, flush)
        self.signature_index = AggregateIndex(self.reload_pack_names, flush)
        all_indices = [self.revision_index, self.inventory_index, self.text_index, self.signature_index]
        if use_chk_index:
            self.chk_index = AggregateIndex(self.reload_pack_names, flush)
            all_indices.append(self.chk_index)
        else:
            self.chk_index = None
        all_combined = [agg_idx.combined_index for agg_idx in all_indices]
        for combined_idx in all_combined:
            combined_idx.set_sibling_indices(set(all_combined).difference([combined_idx]))
        self._resumed_packs = []
        self.config_stack = config.LocationStack(self.transport.base)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.repo)

    def add_pack_to_memory(self, pack):
        """Make a Pack object available to the repository to satisfy queries.

        :param pack: A Pack object.
        """
        if pack.name in self._packs_by_name:
            raise AssertionError('pack {} already in _packs_by_name'.format(pack.name))
        self.packs.append(pack)
        self._packs_by_name[pack.name] = pack
        self.revision_index.add_index(pack.revision_index, pack)
        self.inventory_index.add_index(pack.inventory_index, pack)
        self.text_index.add_index(pack.text_index, pack)
        self.signature_index.add_index(pack.signature_index, pack)
        if self.chk_index is not None:
            self.chk_index.add_index(pack.chk_index, pack)

    def all_packs(self):
        """Return a list of all the Pack objects this repository has.

        Note that an in-progress pack being created is not returned.

        :return: A list of Pack objects for all the packs in the repository.
        """
        result = []
        for name in self.names():
            result.append(self.get_pack_by_name(name))
        return result

    def autopack(self):
        """Pack the pack collection incrementally.

        This will not attempt global reorganisation or recompression,
        rather it will just ensure that the total number of packs does
        not grow without bound. It uses the _max_pack_count method to
        determine if autopacking is needed, and the pack_distribution
        method to determine the number of revisions in each pack.

        If autopacking takes place then the packs name collection will have
        been flushed to disk - packing requires updating the name collection
        in synchronisation with certain steps. Otherwise the names collection
        is not flushed.

        :return: Something evaluating true if packing took place.
        """
        while True:
            try:
                return self._do_autopack()
            except RetryAutopack:
                pass

    def _do_autopack(self):
        total_revisions = self.revision_index.combined_index.key_count()
        total_packs = len(self._names)
        if self._max_pack_count(total_revisions) >= total_packs:
            return None
        pack_distribution = self.pack_distribution(total_revisions)
        existing_packs = []
        for pack in self.all_packs():
            revision_count = pack.get_revision_count()
            if revision_count == 0:
                continue
            existing_packs.append((revision_count, pack))
        pack_operations = self.plan_autopack_combinations(existing_packs, pack_distribution)
        num_new_packs = len(pack_operations)
        num_old_packs = sum([len(po[1]) for po in pack_operations])
        num_revs_affected = sum([po[0] for po in pack_operations])
        mutter('Auto-packing repository %s, which has %d pack files, containing %d revisions. Packing %d files into %d affecting %d revisions', str(self), total_packs, total_revisions, num_old_packs, num_new_packs, num_revs_affected)
        result = self._execute_pack_operations(pack_operations, packer_class=self.normal_packer_class, reload_func=self._restart_autopack)
        mutter('Auto-packing repository %s completed', str(self))
        return result

    def _execute_pack_operations(self, pack_operations, packer_class, reload_func=None):
        """Execute a series of pack operations.

        :param pack_operations: A list of [revision_count, packs_to_combine].
        :param packer_class: The class of packer to use
        :return: The new pack names.
        """
        for revision_count, packs in pack_operations:
            if len(packs) == 0:
                continue
            packer = packer_class(self, packs, '.autopack', reload_func=reload_func)
            try:
                result = packer.pack()
            except RetryWithNewPacks:
                if packer.new_pack is not None:
                    packer.new_pack.abort()
                raise
            if result is None:
                return
            for pack in packs:
                self._remove_pack_from_memory(pack)
        to_be_obsoleted = []
        for _, packs in pack_operations:
            to_be_obsoleted.extend(packs)
        result = self._save_pack_names(clear_obsolete_packs=True, obsolete_packs=to_be_obsoleted)
        return result

    def _flush_new_pack(self):
        if self._new_pack is not None:
            self._new_pack.flush()

    def lock_names(self):
        """Acquire the mutex around the pack-names index.

        This cannot be used in the middle of a read-only transaction on the
        repository.
        """
        self.repo.control_files.lock_write()

    def _already_packed(self):
        """Is the collection already packed?"""
        return not (self.repo._format.pack_compresses or len(self._names) > 1)

    def pack(self, hint=None, clean_obsolete_packs=False):
        """Pack the pack collection totally."""
        self.ensure_loaded()
        total_packs = len(self._names)
        if self._already_packed():
            return
        total_revisions = self.revision_index.combined_index.key_count()
        mutter('Packing repository %s, which has %d pack files, containing %d revisions with hint %r.', str(self), total_packs, total_revisions, hint)
        while True:
            try:
                self._try_pack_operations(hint)
            except RetryPackOperations:
                continue
            break
        if clean_obsolete_packs:
            self._clear_obsolete_packs()

    def _try_pack_operations(self, hint):
        """Calculate the pack operations based on the hint (if any), and
        execute them.
        """
        pack_operations = [[0, []]]
        for pack in self.all_packs():
            if hint is None or pack.name in hint:
                pack_operations[-1][0] += pack.get_revision_count()
                pack_operations[-1][1].append(pack)
        self._execute_pack_operations(pack_operations, packer_class=self.optimising_packer_class, reload_func=self._restart_pack_operations)

    def plan_autopack_combinations(self, existing_packs, pack_distribution):
        """Plan a pack operation.

        :param existing_packs: The packs to pack. (A list of (revcount, Pack)
            tuples).
        :param pack_distribution: A list with the number of revisions desired
            in each pack.
        """
        if len(existing_packs) <= len(pack_distribution):
            return []
        existing_packs.sort(reverse=True)
        pack_operations = [[0, []]]
        while len(existing_packs):
            next_pack_rev_count, next_pack = existing_packs.pop(0)
            if next_pack_rev_count >= pack_distribution[0]:
                while next_pack_rev_count > 0:
                    next_pack_rev_count -= pack_distribution[0]
                    if next_pack_rev_count >= 0:
                        del pack_distribution[0]
                    else:
                        pack_distribution[0] = -next_pack_rev_count
            else:
                pack_operations[-1][0] += next_pack_rev_count
                pack_operations[-1][1].append(next_pack)
                if pack_operations[-1][0] >= pack_distribution[0]:
                    del pack_distribution[0]
                    pack_operations.append([0, []])
        final_rev_count = 0
        final_pack_list = []
        for num_revs, pack_files in pack_operations:
            final_rev_count += num_revs
            final_pack_list.extend(pack_files)
        if len(final_pack_list) == 1:
            raise AssertionError('We somehow generated an autopack with a single pack file being moved.')
            return []
        return [[final_rev_count, final_pack_list]]

    def ensure_loaded(self):
        """Ensure we have read names from disk.

        :return: True if the disk names had not been previously read.
        """
        if not self.repo.is_locked():
            raise errors.ObjectNotLocked(self.repo)
        if self._names is None:
            self._names = {}
            self._packs_at_load = set()
            for index, key, value in self._iter_disk_pack_index():
                name = key[0].decode('ascii')
                self._names[name] = self._parse_index_sizes(value)
                self._packs_at_load.add((name, value))
            result = True
        else:
            result = False
        self.all_packs()
        return result

    def _parse_index_sizes(self, value):
        """Parse a string of index sizes."""
        return tuple((int(digits) for digits in value.split(b' ')))

    def get_pack_by_name(self, name):
        """Get a Pack object by name.

        :param name: The name of the pack - e.g. '123456'
        :return: A Pack object.
        """
        try:
            return self._packs_by_name[name]
        except KeyError:
            rev_index = self._make_index(name, '.rix')
            inv_index = self._make_index(name, '.iix')
            txt_index = self._make_index(name, '.tix')
            sig_index = self._make_index(name, '.six')
            if self.chk_index is not None:
                chk_index = self._make_index(name, '.cix', is_chk=True)
            else:
                chk_index = None
            result = ExistingPack(self._pack_transport, name, rev_index, inv_index, txt_index, sig_index, chk_index)
            self.add_pack_to_memory(result)
            return result

    def _resume_pack(self, name):
        """Get a suspended Pack object by name.

        :param name: The name of the pack - e.g. '123456'
        :return: A Pack object.
        """
        if not re.match('[a-f0-9]{32}', name):
            raise errors.UnresumableWriteGroup(self.repo, [name], 'Malformed write group token')
        try:
            rev_index = self._make_index(name, '.rix', resume=True)
            inv_index = self._make_index(name, '.iix', resume=True)
            txt_index = self._make_index(name, '.tix', resume=True)
            sig_index = self._make_index(name, '.six', resume=True)
            if self.chk_index is not None:
                chk_index = self._make_index(name, '.cix', resume=True, is_chk=True)
            else:
                chk_index = None
            result = self.resumed_pack_factory(name, rev_index, inv_index, txt_index, sig_index, self._upload_transport, self._pack_transport, self._index_transport, self, chk_index=chk_index)
        except _mod_transport.NoSuchFile as e:
            raise errors.UnresumableWriteGroup(self.repo, [name], str(e))
        self.add_pack_to_memory(result)
        self._resumed_packs.append(result)
        return result

    def allocate(self, a_new_pack):
        """Allocate name in the list of packs.

        :param a_new_pack: A NewPack instance to be added to the collection of
            packs for this repository.
        """
        self.ensure_loaded()
        if a_new_pack.name in self._names:
            raise errors.BzrError('Pack {!r} already exists in {}'.format(a_new_pack.name, self))
        self._names[a_new_pack.name] = tuple(a_new_pack.index_sizes)
        self.add_pack_to_memory(a_new_pack)

    def _iter_disk_pack_index(self):
        """Iterate over the contents of the pack-names index.

        This is used when loading the list from disk, and before writing to
        detect updates from others during our write operation.
        :return: An iterator of the index contents.
        """
        return self._index_class(self.transport, 'pack-names', None).iter_all_entries()

    def _make_index(self, name, suffix, resume=False, is_chk=False):
        size_offset = self._suffix_offsets[suffix]
        index_name = name + suffix
        if resume:
            transport = self._upload_transport
            index_size = transport.stat(index_name).st_size
        else:
            transport = self._index_transport
            index_size = self._names[name][size_offset]
        index = self._index_class(transport, index_name, index_size, unlimited_cache=is_chk)
        if is_chk and self._index_class is btree_index.BTreeGraphIndex:
            index._leaf_factory = btree_index._gcchk_factory
        return index

    def _max_pack_count(self, total_revisions):
        """Return the maximum number of packs to use for total revisions.

        :param total_revisions: The total number of revisions in the
            repository.
        """
        if not total_revisions:
            return 1
        digits = str(total_revisions)
        result = 0
        for digit in digits:
            result += int(digit)
        return result

    def names(self):
        """Provide an order to the underlying names."""
        return sorted(self._names.keys())

    def _obsolete_packs(self, packs):
        """Move a number of packs which have been obsoleted out of the way.

        Each pack and its associated indices are moved out of the way.

        Note: for correctness this function should only be called after a new
        pack names index has been written without these pack names, and with
        the names of packs that contain the data previously available via these
        packs.

        :param packs: The packs to obsolete.
        :param return: None.
        """
        for pack in packs:
            try:
                try:
                    pack.pack_transport.move(pack.file_name(), '../obsolete_packs/' + pack.file_name())
                except _mod_transport.NoSuchFile:
                    try:
                        pack.pack_transport.mkdir('../obsolete_packs/')
                    except _mod_transport.FileExists:
                        pass
                    pack.pack_transport.move(pack.file_name(), '../obsolete_packs/' + pack.file_name())
            except (errors.PathError, errors.TransportError) as e:
                mutter("couldn't rename obsolete pack, skipping it:\n%s" % (e,))
            suffixes = ['.iix', '.six', '.tix', '.rix']
            if self.chk_index is not None:
                suffixes.append('.cix')
            for suffix in suffixes:
                try:
                    self._index_transport.move(pack.name + suffix, '../obsolete_packs/' + pack.name + suffix)
                except (errors.PathError, errors.TransportError) as e:
                    mutter("couldn't rename obsolete index, skipping it:\n%s" % (e,))

    def pack_distribution(self, total_revisions):
        """Generate a list of the number of revisions to put in each pack.

        :param total_revisions: The total number of revisions in the
            repository.
        """
        if total_revisions == 0:
            return [0]
        digits = reversed(str(total_revisions))
        result = []
        for exponent, count in enumerate(digits):
            size = 10 ** exponent
            for pos in range(int(count)):
                result.append(size)
        return list(reversed(result))

    def _pack_tuple(self, name):
        """Return a tuple with the transport and file name for a pack name."""
        return (self._pack_transport, name + '.pack')

    def _remove_pack_from_memory(self, pack):
        """Remove pack from the packs accessed by this repository.

        Only affects memory state, until self._save_pack_names() is invoked.
        """
        self._names.pop(pack.name)
        self._packs_by_name.pop(pack.name)
        self._remove_pack_indices(pack)
        self.packs.remove(pack)

    def _remove_pack_indices(self, pack, ignore_missing=False):
        """Remove the indices for pack from the aggregated indices.

        :param ignore_missing: Suppress KeyErrors from calling remove_index.
        """
        for index_type in Pack.index_definitions:
            attr_name = index_type + '_index'
            aggregate_index = getattr(self, attr_name)
            if aggregate_index is not None:
                pack_index = getattr(pack, attr_name)
                try:
                    aggregate_index.remove_index(pack_index)
                except KeyError:
                    if ignore_missing:
                        continue
                    raise

    def reset(self):
        """Clear all cached data."""
        self.revision_index.clear()
        self.signature_index.clear()
        self.text_index.clear()
        self.inventory_index.clear()
        if self.chk_index is not None:
            self.chk_index.clear()
        self._new_pack = None
        self._names = None
        self.packs = []
        self._packs_by_name = {}
        self._packs_at_load = None

    def _unlock_names(self):
        """Release the mutex around the pack-names index."""
        self.repo.control_files.unlock()

    def _diff_pack_names(self):
        """Read the pack names from disk, and compare it to the one in memory.

        :return: (disk_nodes, deleted_nodes, new_nodes)
            disk_nodes    The final set of nodes that should be referenced
            deleted_nodes Nodes which have been removed from when we started
            new_nodes     Nodes that are newly introduced
        """
        disk_nodes = set()
        for index, key, value in self._iter_disk_pack_index():
            disk_nodes.add((key[0].decode('ascii'), value))
        orig_disk_nodes = set(disk_nodes)
        current_nodes = set()
        for name, sizes in self._names.items():
            current_nodes.add((name, b' '.join((b'%d' % size for size in sizes))))
        deleted_nodes = self._packs_at_load - current_nodes
        new_nodes = current_nodes - self._packs_at_load
        disk_nodes.difference_update(deleted_nodes)
        disk_nodes.update(new_nodes)
        return (disk_nodes, deleted_nodes, new_nodes, orig_disk_nodes)

    def _syncronize_pack_names_from_disk_nodes(self, disk_nodes):
        """Given the correct set of pack files, update our saved info.

        :return: (removed, added, modified)
            removed     pack names removed from self._names
            added       pack names added to self._names
            modified    pack names that had changed value
        """
        removed = []
        added = []
        modified = []
        new_names = dict(disk_nodes)
        for pack in self.all_packs():
            if pack.name not in new_names:
                removed.append(pack.name)
                self._remove_pack_from_memory(pack)
        for name, value in disk_nodes:
            sizes = self._parse_index_sizes(value)
            if name in self._names:
                if sizes != self._names[name]:
                    self._remove_pack_from_memory(self.get_pack_by_name(name))
                    self._names[name] = sizes
                    self.get_pack_by_name(name)
                    modified.append(name)
            else:
                self._names[name] = sizes
                self.get_pack_by_name(name)
                added.append(name)
        return (removed, added, modified)

    def _save_pack_names(self, clear_obsolete_packs=False, obsolete_packs=None):
        """Save the list of packs.

        This will take out the mutex around the pack names list for the
        duration of the method call. If concurrent updates have been made, a
        three-way merge between the current list and the current in memory list
        is performed.

        :param clear_obsolete_packs: If True, clear out the contents of the
            obsolete_packs directory.
        :param obsolete_packs: Packs that are obsolete once the new pack-names
            file has been written.
        :return: A list of the names saved that were not previously on disk.
        """
        already_obsolete = []
        self.lock_names()
        try:
            builder = self._index_builder_class()
            disk_nodes, deleted_nodes, new_nodes, orig_disk_nodes = self._diff_pack_names()
            for name, value in disk_nodes:
                builder.add_node((name.encode('ascii'),), value)
            self.transport.put_file('pack-names', builder.finish(), mode=self.repo.controldir._get_file_mode())
            self._packs_at_load = disk_nodes
            if clear_obsolete_packs:
                to_preserve = None
                if obsolete_packs:
                    to_preserve = {o.name for o in obsolete_packs}
                already_obsolete = self._clear_obsolete_packs(to_preserve)
        finally:
            self._unlock_names()
        self._syncronize_pack_names_from_disk_nodes(disk_nodes)
        if obsolete_packs:
            obsolete_packs = [o for o in obsolete_packs if o.name not in already_obsolete]
            self._obsolete_packs(obsolete_packs)
        return [new_node[0] for new_node in new_nodes]

    def reload_pack_names(self):
        """Sync our pack listing with what is present in the repository.

        This should be called when we find out that something we thought was
        present is now missing. This happens when another process re-packs the
        repository, etc.

        :return: True if the in-memory list of packs has been altered at all.
        """
        first_read = self.ensure_loaded()
        if first_read:
            return True
        disk_nodes, deleted_nodes, new_nodes, orig_disk_nodes = self._diff_pack_names()
        self._packs_at_load = orig_disk_nodes
        removed, added, modified = self._syncronize_pack_names_from_disk_nodes(disk_nodes)
        if removed or added or modified:
            return True
        return False

    def _restart_autopack(self):
        """Reload the pack names list, and restart the autopack code."""
        if not self.reload_pack_names():
            raise
        raise RetryAutopack(self.repo, False, sys.exc_info())

    def _restart_pack_operations(self):
        """Reload the pack names list, and restart the autopack code."""
        if not self.reload_pack_names():
            raise
        raise RetryPackOperations(self.repo, False, sys.exc_info())

    def _clear_obsolete_packs(self, preserve=None):
        """Delete everything from the obsolete-packs directory.

        :return: A list of pack identifiers (the filename without '.pack') that
            were found in obsolete_packs.
        """
        found = []
        obsolete_pack_transport = self.transport.clone('obsolete_packs')
        if preserve is None:
            preserve = set()
        try:
            obsolete_pack_files = obsolete_pack_transport.list_dir('.')
        except _mod_transport.NoSuchFile:
            return found
        for filename in obsolete_pack_files:
            name, ext = osutils.splitext(filename)
            if ext == '.pack':
                found.append(name)
            if name in preserve:
                continue
            try:
                obsolete_pack_transport.delete(filename)
            except (errors.PathError, errors.TransportError) as e:
                warning("couldn't delete obsolete pack, skipping it:\n%s" % (e,))
        return found

    def _start_write_group(self):
        if not self.repo.is_write_locked():
            raise errors.NotWriteLocked(self)
        self._new_pack = self.pack_factory(self, upload_suffix='.pack', file_mode=self.repo.controldir._get_file_mode())
        self.revision_index.add_writable_index(self._new_pack.revision_index, self._new_pack)
        self.inventory_index.add_writable_index(self._new_pack.inventory_index, self._new_pack)
        self.text_index.add_writable_index(self._new_pack.text_index, self._new_pack)
        self._new_pack.text_index.set_optimize(combine_backing_indices=False)
        self.signature_index.add_writable_index(self._new_pack.signature_index, self._new_pack)
        if self.chk_index is not None:
            self.chk_index.add_writable_index(self._new_pack.chk_index, self._new_pack)
            self.repo.chk_bytes._index._add_callback = self.chk_index.add_callback
            self._new_pack.chk_index.set_optimize(combine_backing_indices=False)
        self.repo.inventories._index._add_callback = self.inventory_index.add_callback
        self.repo.revisions._index._add_callback = self.revision_index.add_callback
        self.repo.signatures._index._add_callback = self.signature_index.add_callback
        self.repo.texts._index._add_callback = self.text_index.add_callback

    def _abort_write_group(self):
        if self._new_pack is not None:
            with contextlib.ExitStack() as stack:
                stack.callback(setattr, self, '_new_pack', None)
                stack.callback(self._remove_pack_indices, self._new_pack, ignore_missing=True)
                self._new_pack.abort()
        for resumed_pack in self._resumed_packs:
            with contextlib.ExitStack() as stack:
                stack.callback(self._remove_pack_indices, resumed_pack, ignore_missing=True)
                resumed_pack.abort()
        del self._resumed_packs[:]

    def _remove_resumed_pack_indices(self):
        for resumed_pack in self._resumed_packs:
            self._remove_pack_indices(resumed_pack)
        del self._resumed_packs[:]

    def _check_new_inventories(self):
        """Detect missing inventories in this write group.

        :returns: list of strs, summarising any problems found.  If the list is
            empty no problems were found.
        """
        return []

    def _commit_write_group(self):
        all_missing = set()
        for prefix, versioned_file in (('revisions', self.repo.revisions), ('inventories', self.repo.inventories), ('texts', self.repo.texts), ('signatures', self.repo.signatures)):
            missing = versioned_file.get_missing_compression_parent_keys()
            all_missing.update([(prefix,) + key for key in missing])
        if all_missing:
            raise errors.BzrCheckError('Repository %s has missing compression parent(s) %r ' % (self.repo, sorted(all_missing)))
        problems = self._check_new_inventories()
        if problems:
            problems_summary = '\n'.join(problems)
            raise errors.BzrCheckError('Cannot add revision(s) to repository: ' + problems_summary)
        self._remove_pack_indices(self._new_pack)
        any_new_content = False
        if self._new_pack.data_inserted():
            self._new_pack.finish()
            self.allocate(self._new_pack)
            self._new_pack = None
            any_new_content = True
        else:
            self._new_pack.abort()
            self._new_pack = None
        for resumed_pack in self._resumed_packs:
            self._names[resumed_pack.name] = None
            self._remove_pack_from_memory(resumed_pack)
            resumed_pack.finish()
            self.allocate(resumed_pack)
            any_new_content = True
        del self._resumed_packs[:]
        if any_new_content:
            result = self.autopack()
            if not result:
                return self._save_pack_names()
            return result
        return []

    def _suspend_write_group(self):
        tokens = [pack.name for pack in self._resumed_packs]
        self._remove_pack_indices(self._new_pack)
        if self._new_pack.data_inserted():
            self._new_pack.finish(suspend=True)
            tokens.append(self._new_pack.name)
            self._new_pack = None
        else:
            self._new_pack.abort()
            self._new_pack = None
        self._remove_resumed_pack_indices()
        return tokens

    def _resume_write_group(self, tokens):
        for token in tokens:
            self._resume_pack(token)