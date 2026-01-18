import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
class CHKInventoryRepository(PackRepository):
    """subclass of PackRepository that uses CHK based inventories."""

    def __init__(self, _format, a_controldir, control_files, _commit_builder_class, _serializer):
        """Overridden to change pack collection class."""
        super().__init__(_format, a_controldir, control_files, _commit_builder_class, _serializer)
        index_transport = self._transport.clone('indices')
        self._pack_collection = GCRepositoryPackCollection(self, self._transport, index_transport, self._transport.clone('upload'), self._transport.clone('packs'), _format.index_builder_class, _format.index_class, use_chk_index=self._format.supports_chks)
        self.inventories = GroupCompressVersionedFiles(_GCGraphIndex(self._pack_collection.inventory_index.combined_index, add_callback=self._pack_collection.inventory_index.add_callback, parents=True, is_locked=self.is_locked, inconsistency_fatal=False), access=self._pack_collection.inventory_index.data_access)
        self.revisions = GroupCompressVersionedFiles(_GCGraphIndex(self._pack_collection.revision_index.combined_index, add_callback=self._pack_collection.revision_index.add_callback, parents=True, is_locked=self.is_locked, track_external_parent_refs=True, track_new_keys=True), access=self._pack_collection.revision_index.data_access, delta=False)
        self.signatures = GroupCompressVersionedFiles(_GCGraphIndex(self._pack_collection.signature_index.combined_index, add_callback=self._pack_collection.signature_index.add_callback, parents=False, is_locked=self.is_locked, inconsistency_fatal=False), access=self._pack_collection.signature_index.data_access, delta=False)
        self.texts = GroupCompressVersionedFiles(_GCGraphIndex(self._pack_collection.text_index.combined_index, add_callback=self._pack_collection.text_index.add_callback, parents=True, is_locked=self.is_locked, inconsistency_fatal=False), access=self._pack_collection.text_index.data_access)
        self.chk_bytes = GroupCompressVersionedFiles(_GCGraphIndex(self._pack_collection.chk_index.combined_index, add_callback=self._pack_collection.chk_index.add_callback, parents=False, is_locked=self.is_locked, inconsistency_fatal=False), access=self._pack_collection.chk_index.data_access)
        search_key_name = self._format._serializer.search_key_name
        search_key_func = chk_map.search_key_registry.get(search_key_name)
        self.chk_bytes._search_key_func = search_key_func
        self._write_lock_count = 0
        self._transaction = None
        self._reconcile_does_inventory_gc = True
        self._reconcile_fixes_text_parents = True
        self._reconcile_backsup_inventory = False

    def _add_inventory_checked(self, revision_id, inv, parents):
        """Add inv to the repository after checking the inputs.

        This function can be overridden to allow different inventory styles.

        :seealso: add_inventory, for the contract.
        """
        serializer = self._format._serializer
        result = inventory.CHKInventory.from_inventory(self.chk_bytes, inv, maximum_size=serializer.maximum_size, search_key_name=serializer.search_key_name)
        inv_lines = result.to_lines()
        return self._inventory_add_lines(revision_id, parents, inv_lines, check_content=False)

    def _create_inv_from_null(self, delta, revision_id):
        """This will mutate new_inv directly.

        This is a simplified form of create_by_apply_delta which knows that all
        the old values must be None, so everything is a create.
        """
        serializer = self._format._serializer
        new_inv = inventory.CHKInventory(serializer.search_key_name)
        new_inv.revision_id = revision_id
        entry_to_bytes = new_inv._entry_to_bytes
        id_to_entry_dict = {}
        parent_id_basename_dict = {}
        for old_path, new_path, file_id, entry in delta:
            if old_path is not None:
                raise ValueError('Invalid delta, somebody tried to delete %r from the NULL_REVISION' % ((old_path, file_id),))
            if new_path is None:
                raise ValueError('Invalid delta, delta from NULL_REVISION has no new_path %r' % (file_id,))
            if new_path == '':
                new_inv.root_id = file_id
                parent_id_basename_key = StaticTuple(b'', b'').intern()
            else:
                utf8_entry_name = entry.name.encode('utf-8')
                parent_id_basename_key = StaticTuple(entry.parent_id, utf8_entry_name).intern()
            new_value = entry_to_bytes(entry)
            key = StaticTuple(file_id).intern()
            id_to_entry_dict[key] = new_value
            parent_id_basename_dict[parent_id_basename_key] = file_id
        new_inv._populate_from_dicts(self.chk_bytes, id_to_entry_dict, parent_id_basename_dict, maximum_size=serializer.maximum_size)
        return new_inv

    def add_inventory_by_delta(self, basis_revision_id, delta, new_revision_id, parents, basis_inv=None, propagate_caches=False):
        """Add a new inventory expressed as a delta against another revision.

        :param basis_revision_id: The inventory id the delta was created
            against.
        :param delta: The inventory delta (see Inventory.apply_delta for
            details).
        :param new_revision_id: The revision id that the inventory is being
            added for.
        :param parents: The revision ids of the parents that revision_id is
            known to have and are in the repository already. These are supplied
            for repositories that depend on the inventory graph for revision
            graph access, as well as for those that pun ancestry with delta
            compression.
        :param basis_inv: The basis inventory if it is already known,
            otherwise None.
        :param propagate_caches: If True, the caches for this inventory are
          copied to and updated for the result if possible.

        :returns: (validator, new_inv)
            The validator(which is a sha1 digest, though what is sha'd is
            repository format specific) of the serialized inventory, and the
            resulting inventory.
        """
        if not self.is_in_write_group():
            raise AssertionError('{!r} not in write group'.format(self))
        _mod_revision.check_not_reserved_id(new_revision_id)
        basis_tree = None
        if basis_inv is None or not isinstance(basis_inv, inventory.CHKInventory):
            if basis_revision_id == _mod_revision.NULL_REVISION:
                new_inv = self._create_inv_from_null(delta, new_revision_id)
                if new_inv.root_id is None:
                    raise errors.RootMissing()
                inv_lines = new_inv.to_lines()
                return (self._inventory_add_lines(new_revision_id, parents, inv_lines, check_content=False), new_inv)
            else:
                basis_tree = self.revision_tree(basis_revision_id)
                basis_tree.lock_read()
                basis_inv = basis_tree.root_inventory
        try:
            result = basis_inv.create_by_apply_delta(delta, new_revision_id, propagate_caches=propagate_caches)
            inv_lines = result.to_lines()
            return (self._inventory_add_lines(new_revision_id, parents, inv_lines, check_content=False), result)
        finally:
            if basis_tree is not None:
                basis_tree.unlock()

    def _deserialise_inventory(self, revision_id, lines):
        return inventory.CHKInventory.deserialise(self.chk_bytes, lines, (revision_id,))

    def _iter_inventories(self, revision_ids, ordering):
        """Iterate over many inventory objects."""
        if ordering is None:
            ordering = 'unordered'
        keys = [(revision_id,) for revision_id in revision_ids]
        stream = self.inventories.get_record_stream(keys, ordering, True)
        texts = {}
        for record in stream:
            if record.storage_kind != 'absent':
                texts[record.key] = record.get_bytes_as('lines')
            else:
                texts[record.key] = None
        for key in keys:
            lines = texts[key]
            if lines is None:
                yield (None, key[-1])
            else:
                yield (inventory.CHKInventory.deserialise(self.chk_bytes, lines, key), key[-1])

    def _get_inventory_xml(self, revision_id):
        """Get serialized inventory as a string."""
        return self._serializer.write_inventory_to_lines(self.get_inventory(revision_id))

    def _find_present_inventory_keys(self, revision_keys):
        parent_map = self.inventories.get_parent_map(revision_keys)
        present_inventory_keys = {k for k in parent_map}
        return present_inventory_keys

    def fileids_altered_by_revision_ids(self, revision_ids, _inv_weave=None):
        """Find the file ids and versions affected by revisions.

        :param revisions: an iterable containing revision ids.
        :param _inv_weave: The inventory weave from this repository or None.
            If None, the inventory weave will be opened automatically.
        :return: a dictionary mapping altered file-ids to an iterable of
            revision_ids. Each altered file-ids has the exact revision_ids that
            altered it listed explicitly.
        """
        rich_root = self.supports_rich_root()
        bytes_to_info = inventory.CHKInventory._bytes_to_utf8name_key
        file_id_revisions = {}
        with ui.ui_factory.nested_progress_bar() as pb:
            revision_keys = [(r,) for r in revision_ids]
            parent_keys = self._find_parent_keys_of_revisions(revision_keys)
            present_parent_inv_keys = self._find_present_inventory_keys(parent_keys)
            present_parent_inv_ids = {k[-1] for k in present_parent_inv_keys}
            inventories_to_read = set(revision_ids)
            inventories_to_read.update(present_parent_inv_ids)
            root_key_info = _build_interesting_key_sets(self, inventories_to_read, present_parent_inv_ids)
            interesting_root_keys = root_key_info.interesting_root_keys
            uninteresting_root_keys = root_key_info.uninteresting_root_keys
            chk_bytes = self.chk_bytes
            for record, items in chk_map.iter_interesting_nodes(chk_bytes, interesting_root_keys, uninteresting_root_keys, pb=pb):
                for name, bytes in items:
                    name_utf8, file_id, revision_id = bytes_to_info(bytes)
                    if not rich_root and name_utf8 == '':
                        continue
                    try:
                        file_id_revisions[file_id].add(revision_id)
                    except KeyError:
                        file_id_revisions[file_id] = {revision_id}
        return file_id_revisions

    def find_text_key_references(self):
        """Find the text key references within the repository.

        :return: A dictionary mapping text keys ((fileid, revision_id) tuples)
            to whether they were referred to by the inventory of the
            revision_id that they contain. The inventory texts from all present
            revision ids are assessed to generate this report.
        """
        revision_keys = self.revisions.keys()
        result = {}
        rich_roots = self.supports_rich_root()
        with ui.ui_factory.nested_progress_bar() as pb:
            all_revs = self.all_revision_ids()
            total = len(all_revs)
            for pos, inv in enumerate(self.iter_inventories(all_revs)):
                pb.update('Finding text references', pos, total)
                for _, entry in inv.iter_entries():
                    if not rich_roots and entry.file_id == inv.root_id:
                        continue
                    key = (entry.file_id, entry.revision)
                    result.setdefault(key, False)
                    if entry.revision == inv.revision_id:
                        result[key] = True
            return result

    def reconcile_canonicalize_chks(self):
        """Reconcile this repository to make sure all CHKs are in canonical
        form.
        """
        from .reconcile import PackReconciler
        with self.lock_write():
            reconciler = PackReconciler(self, thorough=True, canonicalize_chks=True)
            return reconciler.reconcile()

    def _reconcile_pack(self, collection, packs, extension, revs, pb):
        packer = GCCHKReconcilePacker(collection, packs, extension)
        return packer.pack(pb)

    def _canonicalize_chks_pack(self, collection, packs, extension, revs, pb):
        packer = GCCHKCanonicalizingPacker(collection, packs, extension, revs)
        return packer.pack(pb)

    def _get_source(self, to_format):
        """Return a source for streaming from this repository."""
        if self._format._serializer == to_format._serializer:
            return GroupCHKStreamSource(self, to_format)
        return super()._get_source(to_format)

    def _find_inconsistent_revision_parents(self, revisions_iterator=None):
        """Find revisions with different parent lists in the revision object
        and in the index graph.

        :param revisions_iterator: None, or an iterator of (revid,
            Revision-or-None). This iterator controls the revisions checked.
        :returns: an iterator yielding tuples of (revison-id, parents-in-index,
            parents-in-revision).
        """
        if not self.is_locked():
            raise AssertionError()
        vf = self.revisions
        if revisions_iterator is None:
            revisions_iterator = self.iter_revisions(self.all_revision_ids())
        for revid, revision in revisions_iterator:
            if revision is None:
                pass
            parent_map = vf.get_parent_map([(revid,)])
            parents_according_to_index = tuple((parent[-1] for parent in parent_map[revid,]))
            parents_according_to_revision = tuple(revision.parent_ids)
            if parents_according_to_index != parents_according_to_revision:
                yield (revid, parents_according_to_index, parents_according_to_revision)

    def _check_for_inconsistent_revision_parents(self):
        inconsistencies = list(self._find_inconsistent_revision_parents())
        if inconsistencies:
            raise errors.BzrCheckError('Revision index has inconsistent parents.')