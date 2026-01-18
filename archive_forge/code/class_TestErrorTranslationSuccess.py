import base64
import bz2
import tarfile
import zlib
from io import BytesIO
import fastbencode as bencode
from ... import branch, config, controldir, errors, repository, tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...branch import Branch
from ...revision import NULL_REVISION, Revision
from ...tests import test_server
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from ...transport.remote import (RemoteSSHTransport, RemoteTCPTransport,
from .. import (RemoteBzrProber, bzrdir, groupcompress_repo, inventory,
from ..bzrdir import BzrDir, BzrDirFormat
from ..chk_serializer import chk_bencode_serializer
from ..remote import (RemoteBranch, RemoteBranchFormat, RemoteBzrDir,
from ..smart import medium, request
from ..smart.client import _SmartClient
from ..smart.repository import (SmartServerRepositoryGetParentMap,
class TestErrorTranslationSuccess(TestErrorTranslationBase):
    """Unit tests for breezy.bzr.remote._translate_error.

    Given an ErrorFromSmartServer (which has an error tuple from a smart
    server) and some context, _translate_error raises more specific errors from
    breezy.errors.

    This test case covers the cases where _translate_error succeeds in
    translating an ErrorFromSmartServer to something better.  See
    TestErrorTranslationRobustness for other cases.
    """

    def test_NoSuchRevision(self):
        branch = self.make_branch('')
        revid = b'revid'
        translated_error = self.translateTuple((b'NoSuchRevision', revid), branch=branch)
        expected_error = errors.NoSuchRevision(branch, revid)
        self.assertEqual(expected_error, translated_error)

    def test_nosuchrevision(self):
        repository = self.make_repository('')
        revid = b'revid'
        translated_error = self.translateTuple((b'nosuchrevision', revid), repository=repository)
        expected_error = errors.NoSuchRevision(repository, revid)
        self.assertEqual(expected_error, translated_error)

    def test_nobranch(self):
        bzrdir = self.make_controldir('')
        translated_error = self.translateTuple((b'nobranch',), bzrdir=bzrdir)
        expected_error = errors.NotBranchError(path=bzrdir.root_transport.base)
        self.assertEqual(expected_error, translated_error)

    def test_nobranch_one_arg(self):
        bzrdir = self.make_controldir('')
        translated_error = self.translateTuple((b'nobranch', b'extra detail'), bzrdir=bzrdir)
        expected_error = errors.NotBranchError(path=bzrdir.root_transport.base, detail='extra detail')
        self.assertEqual(expected_error, translated_error)

    def test_norepository(self):
        bzrdir = self.make_controldir('')
        translated_error = self.translateTuple((b'norepository',), bzrdir=bzrdir)
        expected_error = errors.NoRepositoryPresent(bzrdir)
        self.assertEqual(expected_error, translated_error)

    def test_LockContention(self):
        translated_error = self.translateTuple((b'LockContention',))
        expected_error = errors.LockContention('(remote lock)')
        self.assertEqual(expected_error, translated_error)

    def test_UnlockableTransport(self):
        bzrdir = self.make_controldir('')
        translated_error = self.translateTuple((b'UnlockableTransport',), bzrdir=bzrdir)
        expected_error = errors.UnlockableTransport(bzrdir.root_transport)
        self.assertEqual(expected_error, translated_error)

    def test_LockFailed(self):
        lock = 'str() of a server lock'
        why = 'str() of why'
        translated_error = self.translateTuple((b'LockFailed', lock.encode('ascii'), why.encode('ascii')))
        expected_error = errors.LockFailed(lock, why)
        self.assertEqual(expected_error, translated_error)

    def test_TokenMismatch(self):
        token = 'a lock token'
        translated_error = self.translateTuple((b'TokenMismatch',), token=token)
        expected_error = errors.TokenMismatch(token, '(remote token)')
        self.assertEqual(expected_error, translated_error)

    def test_Diverged(self):
        branch = self.make_branch('a')
        other_branch = self.make_branch('b')
        translated_error = self.translateTuple((b'Diverged',), branch=branch, other_branch=other_branch)
        expected_error = errors.DivergedBranches(branch, other_branch)
        self.assertEqual(expected_error, translated_error)

    def test_NotStacked(self):
        branch = self.make_branch('')
        translated_error = self.translateTuple((b'NotStacked',), branch=branch)
        expected_error = errors.NotStacked(branch)
        self.assertEqual(expected_error, translated_error)

    def test_ReadError_no_args(self):
        path = 'a path'
        translated_error = self.translateTuple((b'ReadError',), path=path)
        expected_error = errors.ReadError(path)
        self.assertEqual(expected_error, translated_error)

    def test_ReadError(self):
        path = 'a path'
        translated_error = self.translateTuple((b'ReadError', path.encode('utf-8')))
        expected_error = errors.ReadError(path)
        self.assertEqual(expected_error, translated_error)

    def test_IncompatibleRepositories(self):
        translated_error = self.translateTuple((b'IncompatibleRepositories', b'repo1', b'repo2', b'details here'))
        expected_error = errors.IncompatibleRepositories('repo1', 'repo2', 'details here')
        self.assertEqual(expected_error, translated_error)

    def test_GhostRevisionsHaveNoRevno(self):
        translated_error = self.translateTuple((b'GhostRevisionsHaveNoRevno', b'revid1', b'revid2'))
        expected_error = errors.GhostRevisionsHaveNoRevno(b'revid1', b'revid2')
        self.assertEqual(expected_error, translated_error)

    def test_PermissionDenied_no_args(self):
        path = 'a path'
        translated_error = self.translateTuple((b'PermissionDenied',), path=path)
        expected_error = errors.PermissionDenied(path)
        self.assertEqual(expected_error, translated_error)

    def test_PermissionDenied_one_arg(self):
        path = 'a path'
        translated_error = self.translateTuple((b'PermissionDenied', path.encode('utf-8')))
        expected_error = errors.PermissionDenied(path)
        self.assertEqual(expected_error, translated_error)

    def test_PermissionDenied_one_arg_and_context(self):
        """Given a choice between a path from the local context and a path on
        the wire, _translate_error prefers the path from the local context.
        """
        local_path = 'local path'
        remote_path = 'remote path'
        translated_error = self.translateTuple((b'PermissionDenied', remote_path.encode('utf-8')), path=local_path)
        expected_error = errors.PermissionDenied(local_path)
        self.assertEqual(expected_error, translated_error)

    def test_PermissionDenied_two_args(self):
        path = 'a path'
        extra = 'a string with extra info'
        translated_error = self.translateTuple((b'PermissionDenied', path.encode('utf-8'), extra.encode('utf-8')))
        expected_error = errors.PermissionDenied(path, extra)
        self.assertEqual(expected_error, translated_error)

    def test_NoSuchFile_context_path(self):
        local_path = 'local path'
        translated_error = self.translateTuple((b'ReadError', b'remote path'), path=local_path)
        expected_error = errors.ReadError(local_path)
        self.assertEqual(expected_error, translated_error)

    def test_NoSuchFile_without_context(self):
        remote_path = 'remote path'
        translated_error = self.translateTuple((b'ReadError', remote_path.encode('utf-8')))
        expected_error = errors.ReadError(remote_path)
        self.assertEqual(expected_error, translated_error)

    def test_ReadOnlyError(self):
        translated_error = self.translateTuple((b'ReadOnlyError',))
        expected_error = errors.TransportNotPossible('readonly transport')
        self.assertEqual(expected_error, translated_error)

    def test_MemoryError(self):
        translated_error = self.translateTuple((b'MemoryError',))
        self.assertStartsWith(str(translated_error), 'remote server out of memory')

    def test_generic_IndexError_no_classname(self):
        err = errors.ErrorFromSmartServer((b'error', b'list index out of range'))
        translated_error = self.translateErrorFromSmartServer(err)
        expected_error = UnknownErrorFromSmartServer(err)
        self.assertEqual(expected_error, translated_error)

    def test_generic_KeyError(self):
        err = errors.ErrorFromSmartServer((b'error', b'KeyError', b'1'))
        translated_error = self.translateErrorFromSmartServer(err)
        expected_error = UnknownErrorFromSmartServer(err)
        self.assertEqual(expected_error, translated_error)

    def test_RevnoOutOfBounds(self):
        translated_error = self.translateTuple((b'revno-outofbounds', 5, 0, 3), path=b'path')
        expected_error = errors.RevnoOutOfBounds(5, (0, 3))
        self.assertEqual(expected_error, translated_error)