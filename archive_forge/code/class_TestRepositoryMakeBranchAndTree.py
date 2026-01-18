import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
class TestRepositoryMakeBranchAndTree(per_repository.TestCaseWithRepository):

    def test_repository_format(self):
        tree = self.make_branch_and_tree('repo')
        self.assertIsInstance(tree.branch.repository._format, self.repository_format.__class__)