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
class TestBranchGetTagsBytes(RemoteBranchTestCase):

    def test_backwards_compat(self):
        self.setup_smart_server_with_call_log()
        branch = self.make_branch('.')
        self.reset_smart_call_log()
        verb = b'Branch.get_tags_bytes'
        self.disable_verb(verb)
        branch.tags.get_tag_dict()
        call_count = len([call for call in self.hpss_calls if call.call.method == verb])
        self.assertEqual(1, call_count)

    def test_trivial(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.get_tags_bytes', (b'quack/',), b'success', (b'',))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        result = branch.tags.get_tag_dict()
        self.assertFinished(client)
        self.assertEqual({}, result)