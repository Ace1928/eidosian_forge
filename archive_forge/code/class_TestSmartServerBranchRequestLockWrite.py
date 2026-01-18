import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from breezy import branch as _mod_branch
from breezy import controldir, errors, gpg, tests, transport, urlutils
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import inventory_delta, versionedfile
from breezy.bzr.smart import branch as smart_branch
from breezy.bzr.smart import bzrdir as smart_dir
from breezy.bzr.smart import packrepository as smart_packrepo
from breezy.bzr.smart import repository as smart_repo
from breezy.bzr.smart import request as smart_req
from breezy.bzr.smart import server, vfs
from breezy.bzr.testament import Testament
from breezy.tests import test_server
from breezy.transport import chroot, memory
class TestSmartServerBranchRequestLockWrite(TestLockedBranch):

    def test_lock_write_on_unlocked_branch(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestLockWrite(backing)
        branch = self.make_branch('.', format='knit')
        repository = branch.repository
        response = request.execute(b'')
        branch_nonce = branch.control_files._lock.peek().nonce
        repository_nonce = repository.control_files._lock.peek().nonce
        self.assertEqual(smart_req.SmartServerResponse((b'ok', branch_nonce, repository_nonce)), response)
        new_branch = repository.controldir.open_branch()
        self.assertRaises(errors.LockContention, new_branch.lock_write)
        request = smart_branch.SmartServerBranchRequestUnlock(backing)
        response = request.execute(b'', branch_nonce, repository_nonce)

    def test_lock_write_on_locked_branch(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestLockWrite(backing)
        branch = self.make_branch('.')
        branch_token = branch.lock_write().token
        branch.leave_lock_in_place()
        branch.unlock()
        response = request.execute(b'')
        self.assertEqual(smart_req.SmartServerResponse((b'LockContention',)), response)
        branch.lock_write(branch_token)
        branch.dont_leave_lock_in_place()
        branch.unlock()

    def test_lock_write_with_tokens_on_locked_branch(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestLockWrite(backing)
        branch = self.make_branch('.', format='knit')
        branch_token, repo_token = self.get_lock_tokens(branch)
        branch.leave_lock_in_place()
        branch.repository.leave_lock_in_place()
        branch.unlock()
        response = request.execute(b'', branch_token, repo_token)
        self.assertEqual(smart_req.SmartServerResponse((b'ok', branch_token, repo_token)), response)
        branch.repository.lock_write(repo_token)
        branch.repository.dont_leave_lock_in_place()
        branch.repository.unlock()
        branch.lock_write(branch_token)
        branch.dont_leave_lock_in_place()
        branch.unlock()

    def test_lock_write_with_mismatched_tokens_on_locked_branch(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestLockWrite(backing)
        branch = self.make_branch('.', format='knit')
        branch_token, repo_token = self.get_lock_tokens(branch)
        branch.leave_lock_in_place()
        branch.repository.leave_lock_in_place()
        branch.unlock()
        response = request.execute(b'', branch_token + b'xxx', repo_token)
        self.assertEqual(smart_req.SmartServerResponse((b'TokenMismatch',)), response)
        branch.repository.lock_write(repo_token)
        branch.repository.dont_leave_lock_in_place()
        branch.repository.unlock()
        branch.lock_write(branch_token)
        branch.dont_leave_lock_in_place()
        branch.unlock()

    def test_lock_write_on_locked_repo(self):
        backing = self.get_transport()
        request = smart_branch.SmartServerBranchRequestLockWrite(backing)
        branch = self.make_branch('.', format='knit')
        repo = branch.repository
        repo_token = repo.lock_write().repository_token
        repo.leave_lock_in_place()
        repo.unlock()
        response = request.execute(b'')
        self.assertEqual(smart_req.SmartServerResponse((b'LockContention',)), response)
        repo.lock_write(repo_token)
        repo.dont_leave_lock_in_place()
        repo.unlock()

    def test_lock_write_on_readonly_transport(self):
        backing = self.get_readonly_transport()
        request = smart_branch.SmartServerBranchRequestLockWrite(backing)
        self.make_branch('.')
        root = self.get_transport().clone('/')
        path = urlutils.relative_url(root.base, self.get_transport().base)
        response = request.execute(path.encode('utf-8'))
        error_name, lock_str, why_str = response.args
        self.assertFalse(response.is_successful())
        self.assertEqual(b'LockFailed', error_name)