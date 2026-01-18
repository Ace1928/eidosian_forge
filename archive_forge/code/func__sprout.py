from typing import Any, List
from breezy import branchbuilder
from breezy.branch import GenericInterBranch, InterBranch
from breezy.tests import TestCaseWithTransport, multiply_tests
def _sprout(self, origdir, to_url, format):
    if format.supports_workingtrees:
        newbranch = self.make_branch(to_url, format=format)
    else:
        newbranch = self.make_branch(to_url + '.branch', format=format)
    origbranch = origdir.open_branch()
    newbranch.repository.fetch(origbranch.repository)
    origbranch.copy_content_into(newbranch)
    if format.supports_workingtrees:
        wt = newbranch.controldir.create_workingtree()
    else:
        wt = newbranch.create_checkout(to_url, lightweight=True)
    return wt