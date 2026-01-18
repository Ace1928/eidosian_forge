import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
class TestRemoteGitBranch(TestCaseWithTransport):
    _test_needs_features = [ExecutableFeature('git')]

    def setUp(self):
        TestCaseWithTransport.setUp(self)
        self.remote_real = GitRepo.init('remote', mkdir=True)
        self.remote_url = 'git://%s/' % os.path.abspath(self.remote_real.path)
        self.permit_url(self.remote_url)

    def test_set_last_revision_info(self):
        c1 = self.remote_real.do_commit(message=b'message 1', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/newbranch')
        c2 = self.remote_real.do_commit(message=b'message 2', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/newbranch')
        remote = ControlDir.open(self.remote_url)
        newbranch = remote.open_branch('newbranch')
        self.assertEqual(newbranch.lookup_foreign_revision_id(c2), newbranch.last_revision())
        newbranch.set_last_revision_info(1, newbranch.lookup_foreign_revision_id(c1))
        self.assertEqual(c1, self.remote_real.refs[b'refs/heads/newbranch'])
        self.assertEqual(newbranch.last_revision(), newbranch.lookup_foreign_revision_id(c1))