from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _parent_is_available(self, parent):
    """True if parent is a fully available revision

        A fully available revision has a inventory and a revision object in the
        repository.
        """
    if parent in self._rev_graph:
        return True
    inv_present = 1 == len(self.inventory.get_parent_map([(parent,)]))
    return inv_present and self.repo.has_revision(parent)