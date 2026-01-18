from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
class PackReconciler(VersionedFileRepoReconciler):
    """Reconciler that reconciles a pack based repository.

    Garbage inventories do not affect ancestry queries, and removal is
    considerably more expensive as there is no separate versioned file for
    them, so they are not cleaned. In short it is currently a no-op.

    In future this may be a good place to hook in annotation cache checking,
    index recreation etc.
    """

    def __init__(self, repo, other=None, thorough=False, canonicalize_chks=False):
        super().__init__(repo, other=other, thorough=thorough)
        self.canonicalize_chks = canonicalize_chks

    def _reconcile_steps(self):
        """Perform the steps to reconcile this repository."""
        if not self.thorough:
            return
        collection = self.repo._pack_collection
        collection.ensure_loaded()
        collection.lock_names()
        try:
            packs = collection.all_packs()
            all_revisions = self.repo.all_revision_ids()
            total_inventories = len(list(collection.inventory_index.combined_index.iter_all_entries()))
            if len(all_revisions):
                if self.canonicalize_chks:
                    reconcile_meth = self.repo._canonicalize_chks_pack
                else:
                    reconcile_meth = self.repo._reconcile_pack
                new_pack = reconcile_meth(collection, packs, '.reconcile', all_revisions, self.pb)
                if new_pack is not None:
                    self._discard_and_save(packs)
            else:
                self._discard_and_save(packs)
            self.garbage_inventories = total_inventories - len(list(collection.inventory_index.combined_index.iter_all_entries()))
        finally:
            collection._unlock_names()

    def _discard_and_save(self, packs):
        """Discard some packs from the repository.

        This removes them from the memory index, saves the in-memory index
        which makes the newly reconciled pack visible and hides the packs to be
        discarded, and finally renames the packs being discarded into the
        obsolete packs directory.

        :param packs: The packs to discard.
        """
        for pack in packs:
            self.repo._pack_collection._remove_pack_from_memory(pack)
        self.repo._pack_collection._save_pack_names()
        self.repo._pack_collection._obsolete_packs(packs)