import os
from breezy import branch, errors
from breezy import merge as _mod_merge
from breezy import switch, tests, workingtree
@staticmethod
def _master_if_present(branch):
    master = branch.get_master_branch()
    if master:
        return master
    else:
        return branch