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
class TestBranchHeadsToFetch(RemoteBranchTestCase):

    def test_uses_last_revision_info_and_tags_by_default(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.last_revision_info', (b'quack/',), b'success', (b'ok', b'1', b'rev-tip'))
        client.add_expected_call(b'Branch.get_config_file', (b'quack/',), b'success', (b'ok',), b'')
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        result = branch.heads_to_fetch()
        self.assertFinished(client)
        self.assertEqual(({b'rev-tip'}, set()), result)

    def test_uses_last_revision_info_and_tags_when_set(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.last_revision_info', (b'quack/',), b'success', (b'ok', b'1', b'rev-tip'))
        client.add_expected_call(b'Branch.get_config_file', (b'quack/',), b'success', (b'ok',), b'branch.fetch_tags = True')
        client.add_expected_call(b'Branch.get_tags_bytes', (b'quack/',), b'success', (b'd5:tag-17:rev-foo5:tag-27:rev-bare',))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        result = branch.heads_to_fetch()
        self.assertFinished(client)
        self.assertEqual(({b'rev-tip'}, {b'rev-foo', b'rev-bar'}), result)

    def test_uses_rpc_for_formats_with_non_default_heads_to_fetch(self):
        transport = MemoryTransport()
        client = FakeClient(transport.base)
        client.add_expected_call(b'Branch.get_stacked_on_url', (b'quack/',), b'error', (b'NotStacked',))
        client.add_expected_call(b'Branch.heads_to_fetch', (b'quack/',), b'success', ([b'tip'], [b'tagged-1', b'tagged-2']))
        transport.mkdir('quack')
        transport = transport.clone('quack')
        branch = self.make_remote_branch(transport, client)
        branch._format._use_default_local_heads_to_fetch = lambda: False
        result = branch.heads_to_fetch()
        self.assertFinished(client)
        self.assertEqual(({b'tip'}, {b'tagged-1', b'tagged-2'}), result)

    def make_branch_with_tags(self):
        self.setup_smart_server_with_call_log()
        builder = self.make_branch_builder('foo')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', ''))], revision_id=b'tip')
        builder.finish_series()
        branch = builder.get_branch()
        branch.tags.set_tag('tag-1', b'rev-1')
        branch.tags.set_tag('tag-2', b'rev-2')
        return branch

    def test_backwards_compatible(self):
        br = self.make_branch_with_tags()
        br.get_config_stack().set('branch.fetch_tags', True)
        self.addCleanup(br.lock_read().unlock)
        verb = b'Branch.heads_to_fetch'
        self.disable_verb(verb)
        self.reset_smart_call_log()
        result = br.heads_to_fetch()
        self.assertEqual(({b'tip'}, {b'rev-1', b'rev-2'}), result)
        self.assertEqual([b'Branch.last_revision_info', b'Branch.get_tags_bytes'], [call.call.method for call in self.hpss_calls])

    def test_backwards_compatible_no_tags(self):
        br = self.make_branch_with_tags()
        br.get_config_stack().set('branch.fetch_tags', False)
        self.addCleanup(br.lock_read().unlock)
        verb = b'Branch.heads_to_fetch'
        self.disable_verb(verb)
        self.reset_smart_call_log()
        result = br.heads_to_fetch()
        self.assertEqual(({b'tip'}, set()), result)
        self.assertEqual([b'Branch.last_revision_info'], [call.call.method for call in self.hpss_calls])