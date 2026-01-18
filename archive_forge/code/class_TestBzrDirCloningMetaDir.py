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
class TestBzrDirCloningMetaDir(TestRemote):

    def test_backwards_compat(self):
        self.setup_smart_server_with_call_log()
        a_dir = self.make_controldir('.')
        self.reset_smart_call_log()
        verb = b'BzrDir.cloning_metadir'
        self.disable_verb(verb)
        a_dir.cloning_metadir()
        call_count = len([call for call in self.hpss_calls if call.call.method == verb])
        self.assertEqual(1, call_count)

    def test_branch_reference(self):
        transport = self.get_transport('quack')
        referenced = self.make_branch('referenced')
        expected = referenced.controldir.cloning_metadir()
        client = FakeClient(transport.base)
        (client.add_expected_call(b'BzrDir.cloning_metadir', (b'quack/', b'False'), b'error', (b'BranchReference',)),)
        (client.add_expected_call(b'BzrDir.open_branchV3', (b'quack/',), b'success', (b'ref', self.get_url('referenced').encode('utf-8'))),)
        a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        result = a_controldir.cloning_metadir()
        self.assertEqual(bzrdir.BzrDirMetaFormat1, type(result))
        self.assertEqual(expected._repository_format, result._repository_format)
        self.assertEqual(expected._branch_format, result._branch_format)
        self.assertFinished(client)

    def test_current_server(self):
        transport = self.get_transport('.')
        transport = transport.clone('quack')
        self.make_controldir('quack')
        client = FakeClient(transport.base)
        reference_bzrdir_format = controldir.format_registry.get('default')()
        control_name = reference_bzrdir_format.network_name()
        (client.add_expected_call(b'BzrDir.cloning_metadir', (b'quack/', b'False'), b'success', (control_name, b'', (b'branch', b''))),)
        a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        result = a_controldir.cloning_metadir()
        self.assertEqual(bzrdir.BzrDirMetaFormat1, type(result))
        self.assertEqual(None, result._repository_format)
        self.assertEqual(None, result._branch_format)
        self.assertFinished(client)

    def test_unknown(self):
        transport = self.get_transport('quack')
        referenced = self.make_branch('referenced')
        referenced.controldir.cloning_metadir()
        client = FakeClient(transport.base)
        (client.add_expected_call(b'BzrDir.cloning_metadir', (b'quack/', b'False'), b'success', (b'unknown', b'unknown', (b'branch', b''))),)
        a_controldir = RemoteBzrDir(transport, RemoteBzrDirFormat(), _client=client)
        self.assertRaises(errors.UnknownFormatError, a_controldir.cloning_metadir)