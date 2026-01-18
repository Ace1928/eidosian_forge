from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
class KnitReconciler(VersionedFileRepoReconciler):
    """Reconciler that reconciles a knit format repository.

    This will detect garbage inventories and remove them in thorough mode.
    """

    def _reconcile_steps(self):
        """Perform the steps to reconcile this repository."""
        if self.thorough:
            try:
                self._load_indexes()
            except errors.BzrCheckError:
                self.aborted = True
                return
            self._gc_inventory()
            self._fix_text_parents()

    def _load_indexes(self):
        """Load indexes for the reconciliation."""
        self.transaction = self.repo.get_transaction()
        self.pb.update(gettext('Reading indexes'), 0, 2)
        self.inventory = self.repo.inventories
        self.pb.update(gettext('Reading indexes'), 1, 2)
        self.repo._check_for_inconsistent_revision_parents()
        self.revisions = self.repo.revisions
        self.pb.update(gettext('Reading indexes'), 2, 2)

    def _gc_inventory(self):
        """Remove inventories that are not referenced from the revision store."""
        self.pb.update(gettext('Checking unused inventories'), 0, 1)
        self._check_garbage_inventories()
        self.pb.update(gettext('Checking unused inventories'), 1, 3)
        if not self.garbage_inventories:
            ui.ui_factory.note(gettext('Inventory ok.'))
            return
        self.pb.update(gettext('Backing up inventory'), 0, 0)
        self.repo._backup_inventory()
        ui.ui_factory.note(gettext('Backup Inventory created'))
        new_inventories = self.repo._temp_inventories()
        graph = self.revisions.get_parent_map(self.revisions.keys())
        revision_keys = topo_sort(graph)
        revision_ids = [key[-1] for key in revision_keys]
        self._setup_steps(len(revision_keys))
        stream = self._change_inv_parents(self.inventory.get_record_stream(revision_keys, 'unordered', True), graph.__getitem__, set(revision_keys))
        new_inventories.insert_record_stream(stream)
        if set(new_inventories.keys()) != set(revision_keys):
            raise AssertionError()
        self.pb.update(gettext('Writing weave'))
        self.repo._activate_new_inventory()
        self.inventory = None
        ui.ui_factory.note(gettext('Inventory regenerated.'))

    def _fix_text_parents(self):
        """Fix bad versionedfile parent entries.

        It is possible for the parents entry in a versionedfile entry to be
        inconsistent with the values in the revision and inventory.

        This method finds entries with such inconsistencies, corrects their
        parent lists, and replaces the versionedfile with a corrected version.
        """
        transaction = self.repo.get_transaction()
        versions = [key[-1] for key in self.revisions.keys()]
        mutter('Prepopulating revision text cache with %d revisions', len(versions))
        vf_checker = self.repo._get_versioned_file_checker()
        bad_parents, unused_versions = vf_checker.check_file_version_parents(self.repo.texts, self.pb)
        text_index = vf_checker.text_index
        per_id_bad_parents = {}
        for key in unused_versions:
            per_id_bad_parents[key[0]] = {}
        for key, details in bad_parents.items():
            file_id = key[0]
            rev_id = key[1]
            knit_parents = tuple([parent[-1] for parent in details[0]])
            correct_parents = tuple([parent[-1] for parent in details[1]])
            file_details = per_id_bad_parents.setdefault(file_id, {})
            file_details[rev_id] = (knit_parents, correct_parents)
        file_id_versions = {}
        for text_key in text_index:
            versions_list = file_id_versions.setdefault(text_key[0], [])
            versions_list.append(text_key[1])
        for num, file_id in enumerate(per_id_bad_parents):
            self.pb.update(gettext('Fixing text parents'), num, len(per_id_bad_parents))
            versions_with_bad_parents = per_id_bad_parents[file_id]
            id_unused_versions = {key[-1] for key in unused_versions if key[0] == file_id}
            if file_id in file_id_versions:
                file_versions = file_id_versions[file_id]
            else:
                file_versions = []
            self._fix_text_parent(file_id, versions_with_bad_parents, id_unused_versions, file_versions)

    def _fix_text_parent(self, file_id, versions_with_bad_parents, unused_versions, all_versions):
        """Fix bad versionedfile entries in a single versioned file."""
        mutter('fixing text parent: %r (%d versions)', file_id, len(versions_with_bad_parents))
        mutter('(%d are unused)', len(unused_versions))
        new_file_id = b'temp:%s' % file_id
        new_parents = {}
        needed_keys = set()
        for version in all_versions:
            if version in unused_versions:
                continue
            elif version in versions_with_bad_parents:
                parents = versions_with_bad_parents[version][1]
            else:
                pmap = self.repo.texts.get_parent_map([(file_id, version)])
                parents = [key[-1] for key in pmap[file_id, version]]
            new_parents[new_file_id, version] = [(new_file_id, parent) for parent in parents]
            needed_keys.add((file_id, version))

        def fix_parents(stream):
            for record in stream:
                chunks = record.get_bytes_as('chunked')
                new_key = (new_file_id, record.key[-1])
                parents = new_parents[new_key]
                yield ChunkedContentFactory(new_key, parents, record.sha1, chunks)
        stream = self.repo.texts.get_record_stream(needed_keys, 'topological', True)
        self.repo._remove_file_id(new_file_id)
        self.repo.texts.insert_record_stream(fix_parents(stream))
        self.repo._remove_file_id(file_id)
        if len(new_parents):
            self.repo._move_file_id(new_file_id, file_id)