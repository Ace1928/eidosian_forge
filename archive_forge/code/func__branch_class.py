from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def _branch_class(self):
    return TrimmedBranch