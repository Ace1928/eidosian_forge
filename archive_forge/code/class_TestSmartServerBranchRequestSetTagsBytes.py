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
class TestSmartServerBranchRequestSetTagsBytes(TestLockedBranch):

    def test_set_bytes(self):
        base_branch = self.make_branch('base')
        tag_bytes = base_branch._get_tags_bytes()
        branch_token, repo_token = self.get_lock_tokens(base_branch)
        request = smart_branch.SmartServerBranchSetTagsBytes(self.get_transport())
        response = request.execute(b'base', branch_token, repo_token)
        self.assertEqual(None, response)
        response = request.do_chunk(tag_bytes)
        self.assertEqual(None, response)
        response = request.do_end()
        self.assertEqual(smart_req.SuccessfulSmartServerResponse(()), response)
        base_branch.unlock()

    def test_lock_failed(self):
        base_branch = self.make_branch('base')
        base_branch.lock_write()
        tag_bytes = base_branch._get_tags_bytes()
        request = smart_branch.SmartServerBranchSetTagsBytes(self.get_transport())
        self.assertRaises(errors.TokenMismatch, request.execute, b'base', b'wrong token', b'wrong token')
        request.do_chunk(tag_bytes)
        request.do_end()
        base_branch.unlock()