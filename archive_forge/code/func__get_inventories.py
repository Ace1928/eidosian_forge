from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _get_inventories(self, revision_ids):
    """Get the inventories for revision-ids.

        This is a callback used by the RepositoryStore to
        speed up inventory reconstruction.
        """
    present = []
    inventories = []
    for revision_id in revision_ids:
        try:
            inv = self.cache_mgr.inventories[revision_id]
            present.append(revision_id)
        except KeyError:
            if self.verbose:
                self.note('get_inventories cache miss for %s', revision_id)
            try:
                inv = self.get_inventory(revision_id)
                present.append(revision_id)
            except:
                inv = self._init_inventory()
            self.cache_mgr.inventories[revision_id] = inv
        inventories.append(inv)
    return (present, inventories)