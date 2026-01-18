from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _reweave_inventory(self):
    """Regenerate the inventory weave for the repository from scratch.

        This is a smart function: it will only do the reweave if doing it
        will correct data issues. The self.thorough flag controls whether
        only data-loss causing issues (!self.thorough) or all issues
        (self.thorough) are treated as requiring the reweave.
        """
    transaction = self.repo.get_transaction()
    self.pb.update(gettext('Reading inventory data'))
    self.inventory = self.repo.inventories
    self.revisions = self.repo.revisions
    self.pending = {key[-1] for key in self.revisions.keys()}
    self._rev_graph = {}
    self.inconsistent_parents = 0
    self._setup_steps(len(self.pending))
    for rev_id in self.pending:
        self._graph_revision(rev_id)
    self._check_garbage_inventories()
    if not self.inconsistent_parents and (not self.garbage_inventories or not self.thorough):
        ui.ui_factory.note(gettext('Inventory ok.'))
        return
    self.pb.update(gettext('Backing up inventory'), 0, 0)
    self.repo._backup_inventory()
    ui.ui_factory.note(gettext('Backup inventory created.'))
    new_inventories = self.repo._temp_inventories()
    self._setup_steps(len(self._rev_graph))
    revision_keys = [(rev_id,) for rev_id in topo_sort(self._rev_graph)]
    stream = self._change_inv_parents(self.inventory.get_record_stream(revision_keys, 'unordered', True), self._new_inv_parents, set(revision_keys))
    new_inventories.insert_record_stream(stream)
    if not set(new_inventories.keys()) == {(revid,) for revid in self.pending}:
        raise AssertionError()
    self.pb.update(gettext('Writing weave'))
    self.repo._activate_new_inventory()
    self.inventory = None
    ui.ui_factory.note(gettext('Inventory regenerated.'))