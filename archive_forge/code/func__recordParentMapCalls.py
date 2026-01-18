from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
def _recordParentMapCalls(self, repo):
    self._parent_map_calls = []
    orig_get_parent_map = repo.revisions.get_parent_map

    def get_parent_map(q):
        q = list(q)
        self._parent_map_calls.extend([e[0] for e in q])
        return orig_get_parent_map(q)
    repo.revisions.get_parent_map = get_parent_map