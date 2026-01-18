import os
import tempfile
from io import BytesIO
from itertools import chain
from ...objects import hex_to_sha
from ...repo import Repo, check_ref_format
from .utils import CompatTestCase, require_git_version, rmtree_ro, run_git_or_fail
class InitNewWorkingDirectoryTestCase(WorkingTreeTestCase):
    """Test compatibility of Repo.init_new_working_directory."""
    min_git_version = (2, 5, 0)

    def setUp(self):
        super().setUp()
        self._other_worktree = self._repo
        worktree_repo_path = tempfile.mkdtemp()
        self.addCleanup(rmtree_ro, worktree_repo_path)
        self._repo = Repo._init_new_working_directory(worktree_repo_path, self._mainworktree_repo)
        self.addCleanup(self._repo.close)
        self._number_of_working_tree = 3

    def test_head_equality(self):
        self.assertEqual(self._repo.refs[b'HEAD'], self._mainworktree_repo.refs[b'HEAD'])

    def test_bare(self):
        self.assertFalse(self._repo.bare)
        self.assertTrue(os.path.isfile(os.path.join(self._repo.path, '.git')))