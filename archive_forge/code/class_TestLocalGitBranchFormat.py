import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
class TestLocalGitBranchFormat(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.format = branch.LocalGitBranchFormat()

    def test_get_format_description(self):
        self.assertEqual('Local Git Branch', self.format.get_format_description())

    def test_get_network_name(self):
        self.assertEqual(b'git', self.format.network_name())

    def test_supports_tags(self):
        self.assertTrue(self.format.supports_tags())