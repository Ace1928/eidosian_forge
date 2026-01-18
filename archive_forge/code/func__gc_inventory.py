from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
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