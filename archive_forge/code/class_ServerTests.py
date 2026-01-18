import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
class ServerTests:
    """Base tests for testing servers.

    Does not inherit from TestCase so tests are not automatically run.
    """
    min_single_branch_version = (1, 7, 10)

    def import_repos(self):
        self._old_repo = self.import_repo('server_old.export')
        self._new_repo = self.import_repo('server_new.export')

    def url(self, port):
        return f'{self.protocol}://localhost:{port}/'

    def branch_args(self, branches=None):
        if branches is None:
            branches = ['master', 'branch']
        return [f'{b}:{b}' for b in branches]

    def test_push_to_dulwich(self):
        self.import_repos()
        self.assertReposNotEqual(self._old_repo, self._new_repo)
        port = self._start_server(self._old_repo)
        run_git_or_fail(['push', self.url(port), *self.branch_args()], cwd=self._new_repo.path)
        self.assertReposEqual(self._old_repo, self._new_repo)

    def test_push_to_dulwich_no_op(self):
        self._old_repo = self.import_repo('server_old.export')
        self._new_repo = self.import_repo('server_old.export')
        self.assertReposEqual(self._old_repo, self._new_repo)
        port = self._start_server(self._old_repo)
        run_git_or_fail(['push', self.url(port), *self.branch_args()], cwd=self._new_repo.path)
        self.assertReposEqual(self._old_repo, self._new_repo)

    def test_push_to_dulwich_remove_branch(self):
        self._old_repo = self.import_repo('server_old.export')
        self._new_repo = self.import_repo('server_old.export')
        self.assertReposEqual(self._old_repo, self._new_repo)
        port = self._start_server(self._old_repo)
        run_git_or_fail(['push', self.url(port), ':master'], cwd=self._new_repo.path)
        self.assertEqual(list(self._old_repo.get_refs().keys()), [b'refs/heads/branch'])

    def test_fetch_from_dulwich(self):
        self.import_repos()
        self.assertReposNotEqual(self._old_repo, self._new_repo)
        port = self._start_server(self._new_repo)
        run_git_or_fail(['fetch', self.url(port), *self.branch_args()], cwd=self._old_repo.path)
        self._old_repo.object_store._pack_cache_time = 0
        self.assertReposEqual(self._old_repo, self._new_repo)

    def test_fetch_from_dulwich_no_op(self):
        self._old_repo = self.import_repo('server_old.export')
        self._new_repo = self.import_repo('server_old.export')
        self.assertReposEqual(self._old_repo, self._new_repo)
        port = self._start_server(self._new_repo)
        run_git_or_fail(['fetch', self.url(port), *self.branch_args()], cwd=self._old_repo.path)
        self._old_repo.object_store._pack_cache_time = 0
        self.assertReposEqual(self._old_repo, self._new_repo)

    def test_clone_from_dulwich_empty(self):
        old_repo_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, old_repo_dir)
        self._old_repo = Repo.init_bare(old_repo_dir)
        port = self._start_server(self._old_repo)
        new_repo_base_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, new_repo_base_dir)
        new_repo_dir = os.path.join(new_repo_base_dir, 'empty_new')
        run_git_or_fail(['clone', self.url(port), new_repo_dir], cwd=new_repo_base_dir)
        new_repo = Repo(new_repo_dir)
        self.assertReposEqual(self._old_repo, new_repo)

    def test_lsremote_from_dulwich(self):
        self._repo = self.import_repo('server_old.export')
        port = self._start_server(self._repo)
        o = run_git_or_fail(['ls-remote', self.url(port)])
        self.assertEqual(len(o.split(b'\n')), 4)

    def test_new_shallow_clone_from_dulwich(self):
        require_git_version(self.min_single_branch_version)
        self._source_repo = self.import_repo('server_new.export')
        self._stub_repo = _StubRepo('shallow')
        self.addCleanup(tear_down_repo, self._stub_repo)
        port = self._start_server(self._source_repo)
        run_git_or_fail(['clone', '--mirror', '--depth=1', '--no-single-branch', self.url(port), self._stub_repo.path])
        clone = self._stub_repo = Repo(self._stub_repo.path)
        expected_shallow = [b'35e0b59e187dd72a0af294aedffc213eaa4d03ff', b'514dc6d3fbfe77361bcaef320c4d21b72bc10be9']
        self.assertEqual(expected_shallow, _get_shallow(clone))
        self.assertReposNotEqual(clone, self._source_repo)

    def test_shallow_clone_from_git_is_identical(self):
        require_git_version(self.min_single_branch_version)
        self._source_repo = self.import_repo('server_new.export')
        self._stub_repo_git = _StubRepo('shallow-git')
        self.addCleanup(tear_down_repo, self._stub_repo_git)
        self._stub_repo_dw = _StubRepo('shallow-dw')
        self.addCleanup(tear_down_repo, self._stub_repo_dw)
        run_git_or_fail(['clone', '--mirror', '--depth=1', '--no-single-branch', 'file://' + self._source_repo.path, self._stub_repo_git.path])
        port = self._start_server(self._source_repo)
        run_git_or_fail(['clone', '--mirror', '--depth=1', '--no-single-branch', self.url(port), self._stub_repo_dw.path])
        self.assertReposEqual(Repo(self._stub_repo_git.path), Repo(self._stub_repo_dw.path))

    def test_fetch_same_depth_into_shallow_clone_from_dulwich(self):
        require_git_version(self.min_single_branch_version)
        self._source_repo = self.import_repo('server_new.export')
        self._stub_repo = _StubRepo('shallow')
        self.addCleanup(tear_down_repo, self._stub_repo)
        port = self._start_server(self._source_repo)
        run_git_or_fail(['clone', '--mirror', '--depth=2', '--no-single-branch', self.url(port), self._stub_repo.path])
        clone = self._stub_repo = Repo(self._stub_repo.path)
        run_git_or_fail(['fetch', '--depth=2', self.url(port), *self.branch_args()], cwd=self._stub_repo.path)
        expected_shallow = [b'94de09a530df27ac3bb613aaecdd539e0a0655e1', b'da5cd81e1883c62a25bb37c4d1f8ad965b29bf8d']
        self.assertEqual(expected_shallow, _get_shallow(clone))
        self.assertReposNotEqual(clone, self._source_repo)

    def test_fetch_full_depth_into_shallow_clone_from_dulwich(self):
        require_git_version(self.min_single_branch_version)
        self._source_repo = self.import_repo('server_new.export')
        self._stub_repo = _StubRepo('shallow')
        self.addCleanup(tear_down_repo, self._stub_repo)
        port = self._start_server(self._source_repo)
        run_git_or_fail(['clone', '--mirror', '--depth=2', '--no-single-branch', self.url(port), self._stub_repo.path])
        clone = self._stub_repo = Repo(self._stub_repo.path)
        run_git_or_fail(['fetch', '--depth=2', self.url(port), *self.branch_args()], cwd=self._stub_repo.path)
        run_git_or_fail(['fetch', '--depth=4', self.url(port), *self.branch_args()], cwd=self._stub_repo.path)
        self.assertEqual([], _get_shallow(clone))
        self.assertReposEqual(clone, self._source_repo)

    def test_fetch_from_dulwich_issue_88_standard(self):
        self._source_repo = self.import_repo('issue88_expect_ack_nak_server.export')
        self._client_repo = self.import_repo('issue88_expect_ack_nak_client.export')
        port = self._start_server(self._source_repo)
        run_git_or_fail(['fetch', self.url(port), 'master'], cwd=self._client_repo.path)
        self.assertObjectStoreEqual(self._source_repo.object_store, self._client_repo.object_store)

    def test_fetch_from_dulwich_issue_88_alternative(self):
        self._source_repo = self.import_repo('issue88_expect_ack_nak_other.export')
        self._client_repo = self.import_repo('issue88_expect_ack_nak_client.export')
        port = self._start_server(self._source_repo)
        self.assertRaises(KeyError, self._client_repo.get_object, b'02a14da1fc1fc13389bbf32f0af7d8899f2b2323')
        run_git_or_fail(['fetch', self.url(port), 'master'], cwd=self._client_repo.path)
        self.assertEqual(b'commit', self._client_repo.get_object(b'02a14da1fc1fc13389bbf32f0af7d8899f2b2323').type_name)

    def test_push_to_dulwich_issue_88_standard(self):
        self._source_repo = self.import_repo('issue88_expect_ack_nak_client.export')
        self._client_repo = self.import_repo('issue88_expect_ack_nak_server.export')
        port = self._start_server(self._source_repo)
        run_git_or_fail(['push', self.url(port), 'master'], cwd=self._client_repo.path)
        self.assertReposEqual(self._source_repo, self._client_repo)