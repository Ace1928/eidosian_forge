import time
import configobj
from fastimport import commands
from fastimport import errors as plugin_errors
from fastimport import processor
from fastimport.helpers import invert_dictset
from .... import debug, delta, errors, osutils, progress
from .... import revision as _mod_revision
from ....bzr.knitpack_repo import KnitPackRepository
from ....trace import mutter, note, warning
from .. import (branch_updater, cache_manager, helpers, idmapfile, marks_file,
def _get_working_trees(self, branches):
    """Get the working trees for branches in the repository."""
    result = []
    wt_expected = self.repo.make_working_trees()
    for br in branches:
        if br is None:
            continue
        elif br == self.branch:
            if self.working_tree:
                result.append(self.working_tree)
        elif wt_expected:
            try:
                result.append(br.controldir.open_workingtree())
            except errors.NoWorkingTree:
                self.warning('No working tree for branch %s', br)
    return result