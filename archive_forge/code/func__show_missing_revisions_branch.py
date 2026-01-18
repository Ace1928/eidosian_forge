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
def _show_missing_revisions_branch(branch, outfile):
    """Show missing master revisions in branch."""
    master = branch.get_master_branch()
    if master:
        local_extra, remote_extra = find_unmerged(branch, master)
        if remote_extra:
            outfile.write('\n')
            outfile.write('Branch is out of date: missing %d revision%s.\n' % (len(remote_extra), plural(len(remote_extra))))