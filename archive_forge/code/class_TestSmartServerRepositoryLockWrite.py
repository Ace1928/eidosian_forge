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
class TestSmartServerRepositoryLockWrite(tests.TestCaseWithMemoryTransport):

    def test_lock_write_on_unlocked_repo(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryLockWrite(backing)
        repository = self.make_repository('.', format='knit')
        response = request.execute(b'')
        nonce = repository.control_files._lock.peek().nonce
        self.assertEqual(smart_req.SmartServerResponse((b'ok', nonce)), response)
        new_repo = repository.controldir.open_repository()
        self.assertRaises(errors.LockContention, new_repo.lock_write)
        request = smart_repo.SmartServerRepositoryUnlock(backing)
        response = request.execute(b'', nonce)

    def test_lock_write_on_locked_repo(self):
        backing = self.get_transport()
        request = smart_repo.SmartServerRepositoryLockWrite(backing)
        repository = self.make_repository('.', format='knit')
        repo_token = repository.lock_write().repository_token
        repository.leave_lock_in_place()
        repository.unlock()
        response = request.execute(b'')
        self.assertEqual(smart_req.SmartServerResponse((b'LockContention',)), response)
        repository.lock_write(repo_token)
        repository.dont_leave_lock_in_place()
        repository.unlock()

    def test_lock_write_on_readonly_transport(self):
        backing = self.get_readonly_transport()
        request = smart_repo.SmartServerRepositoryLockWrite(backing)
        self.make_repository('.', format='knit')
        response = request.execute(b'')
        self.assertFalse(response.is_successful())
        self.assertEqual(b'LockFailed', response.args[0])