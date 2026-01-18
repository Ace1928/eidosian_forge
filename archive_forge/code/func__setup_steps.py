from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _setup_steps(self, new_total):
    """Setup the markers we need to control the progress bar."""
    self.total = new_total
    self.count = 0