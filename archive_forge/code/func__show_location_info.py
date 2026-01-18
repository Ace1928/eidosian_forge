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
def _show_location_info(locs, outfile):
    """Show known locations for working, branch and repository."""
    outfile.write('Location:\n')
    path_list = LocationList(osutils.getcwd())
    for name, loc in locs:
        path_list.add_url(name, loc)
    outfile.writelines(path_list.get_lines())