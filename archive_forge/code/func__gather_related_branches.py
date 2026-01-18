import sys
import time
from io import StringIO
from . import branch as _mod_branch
from . import controldir, errors
from . import hooks as _mod_hooks
from . import osutils, urlutils
from .bzr import bzrdir
from .errors import (NoRepositoryPresent, NotBranchError, NotLocalUrl,
from .missing import find_unmerged
def _gather_related_branches(branch):
    locs = LocationList(osutils.getcwd())
    locs.add_url('public branch', branch.get_public_branch())
    locs.add_url('push branch', branch.get_push_location())
    locs.add_url('parent branch', branch.get_parent())
    locs.add_url('submit branch', branch.get_submit_branch())
    try:
        locs.add_url('stacked on', branch.get_stacked_on_url())
    except (_mod_branch.UnstackableBranchFormat, errors.UnstackableRepositoryFormat, errors.NotStacked):
        pass
    return locs