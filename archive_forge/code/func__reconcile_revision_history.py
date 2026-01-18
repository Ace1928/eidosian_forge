from .. import errors
from .. import revision as _mod_revision
from .. import ui
from ..i18n import gettext
from ..reconcile import ReconcileResult
from ..trace import mutter
from ..tsort import topo_sort
from .versionedfile import AdapterFactory, ChunkedContentFactory
def _reconcile_revision_history(self):
    last_revno, last_revision_id = self.branch.last_revision_info()
    real_history = []
    graph = self.branch.repository.get_graph()
    try:
        for revid in graph.iter_lefthand_ancestry(last_revision_id, (_mod_revision.NULL_REVISION,)):
            real_history.append(revid)
    except errors.RevisionNotPresent:
        pass
    real_history.reverse()
    if last_revno != len(real_history):
        ui.ui_factory.note(gettext('Fixing last revision info {0}  => {1}').format(last_revno, len(real_history)))
        self.branch.set_last_revision_info(len(real_history), last_revision_id)
        return True
    else:
        ui.ui_factory.note(gettext('revision_history ok.'))
        return False