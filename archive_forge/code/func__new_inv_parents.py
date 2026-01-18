from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _new_inv_parents(self, revision_key):
    """Lookup ghost-filtered parents for revision_key."""
    return tuple([(revid,) for revid in self._rev_graph[revision_key[-1]]])