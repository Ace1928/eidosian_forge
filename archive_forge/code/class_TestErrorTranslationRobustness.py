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
class TestErrorTranslationRobustness(TestErrorTranslationBase):
    """Unit tests for breezy.bzr.remote._translate_error's robustness.

    TestErrorTranslationSuccess is for cases where _translate_error can
    translate successfully.  This class about how _translate_err behaves when
    it fails to translate: it re-raises the original error.
    """

    def test_unrecognised_server_error(self):
        """If the error code from the server is not recognised, the original
        ErrorFromSmartServer is propagated unmodified.
        """
        error_tuple = (b'An unknown error tuple',)
        server_error = errors.ErrorFromSmartServer(error_tuple)
        translated_error = self.translateErrorFromSmartServer(server_error)
        expected_error = UnknownErrorFromSmartServer(server_error)
        self.assertEqual(expected_error, translated_error)

    def test_context_missing_a_key(self):
        """In case of a bug in the client, or perhaps an unexpected response
        from a server, _translate_error returns the original error tuple from
        the server and mutters a warning.
        """
        error_tuple = (b'NoSuchRevision', b'revid')
        server_error = errors.ErrorFromSmartServer(error_tuple)
        translated_error = self.translateErrorFromSmartServer(server_error)
        self.assertEqual(server_error, translated_error)
        self.assertContainsRe(self.get_log(), "Missing key 'branch' in context")

    def test_path_missing(self):
        """Some translations (PermissionDenied, ReadError) can determine the
        'path' variable from either the wire or the local context.  If neither
        has it, then an error is raised.
        """
        error_tuple = (b'ReadError',)
        server_error = errors.ErrorFromSmartServer(error_tuple)
        translated_error = self.translateErrorFromSmartServer(server_error)
        self.assertEqual(server_error, translated_error)
        self.assertContainsRe(self.get_log(), "Missing key 'path' in context")