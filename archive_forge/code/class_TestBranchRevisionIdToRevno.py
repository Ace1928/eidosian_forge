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
class TestBranchRevisionIdToRevno(RemoteBranchTestCase):

    def test_simple(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.revision_id_to_revno', (b'quack/', b'null:'), b'success', (b'ok', b'0'))
        client.add_expected_call(b'Branch.revision_id_to_revno', (b'quack/', b'unknown'), b'error', (b'NoSuchRevision', b'unknown'))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        self.assertEqual(0, branch.revision_id_to_revno(b'null:'))
        self.assertRaises(errors.NoSuchRevision, branch.revision_id_to_revno, b'unknown')
        self.assertFinished(client)

    def test_dotted(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.revision_id_to_revno', (b'quack/', b'null:'), b'success', (b'ok', b'0'))
        client.add_expected_call(b'Branch.revision_id_to_revno', (b'quack/', b'unknown'), b'error', (b'NoSuchRevision', b'unknown'))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        self.assertEqual((0,), branch.revision_id_to_dotted_revno(b'null:'))
        self.assertRaises(errors.NoSuchRevision, branch.revision_id_to_dotted_revno, b'unknown')
        self.assertFinished(client)

    def test_ghost_revid(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.revision_id_to_revno', (b'quack/', b'revid'), b'error', (b'error', b'GhostRevisionsHaveNoRevno', b'The reivison {revid} was not found because there was a ghost at {ghost-revid}'))
        client.add_expected_call(b'Branch.revision_id_to_revno', (b'quack/', b'revid'), b'error', (b'GhostRevisionsHaveNoRevno', b'revid', b'ghost-revid'))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        self.assertRaises(errors.GhostRevisionsHaveNoRevno, branch.revision_id_to_dotted_revno, b'revid')
        self.assertRaises(errors.GhostRevisionsHaveNoRevno, branch.revision_id_to_dotted_revno, b'revid')
        self.assertFinished(client)

    def test_dotted_no_smart_verb(self):
        self.setup_smart_server_with_call_log()
        branch = self.make_branch('.')
        self.disable_verb(b'Branch.revision_id_to_revno')
        self.reset_smart_call_log()
        self.assertEqual((0,), branch.revision_id_to_dotted_revno(b'null:'))
        self.assertLength(8, self.hpss_calls)