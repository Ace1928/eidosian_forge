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
class TestRepositoryInsertStream(TestRepositoryInsertStreamBase):
    """Tests for using Repository.insert_stream verb when the _1.19 variant is
    not available.

    This test case is very similar to TestRepositoryInsertStream_1_19.
    """

    def setUp(self):
        super().setUp()
        self.disable_verb(b'Repository.insert_stream_1.19')

    def test_unlocked_repo(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.insert_stream_1.19', (b'quack/', b''), b'unknown', (b'Repository.insert_stream_1.19',))
        client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))
        client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))
        self.checkInsertEmptyStream(repo, client)

    def test_locked_repo_with_no_lock_token(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b''))
        client.add_expected_call(b'Repository.insert_stream_1.19', (b'quack/', b''), b'unknown', (b'Repository.insert_stream_1.19',))
        client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))
        client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))
        repo.lock_write()
        self.checkInsertEmptyStream(repo, client)

    def test_locked_repo_with_lock_token(self):
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.lock_write', (b'quack/', b''), b'success', (b'ok', b'a token'))
        client.add_expected_call(b'Repository.insert_stream_1.19', (b'quack/', b'', b'a token'), b'unknown', (b'Repository.insert_stream_1.19',))
        client.add_expected_call(b'Repository.insert_stream_locked', (b'quack/', b'', b'a token'), b'success', (b'ok',))
        client.add_expected_call(b'Repository.insert_stream_locked', (b'quack/', b'', b'a token'), b'success', (b'ok',))
        repo.lock_write()
        self.checkInsertEmptyStream(repo, client)

    def test_stream_with_inventory_deltas(self):
        """'inventory-deltas' substreams cannot be sent to the
        Repository.insert_stream verb, because not all servers that implement
        that verb will accept them.  So when one is encountered the RemoteSink
        immediately stops using that verb and falls back to VFS insert_stream.
        """
        transport_path = 'quack'
        repo, client = self.setup_fake_client_and_repository(transport_path)
        client.add_expected_call(b'Repository.insert_stream_1.19', (b'quack/', b''), b'unknown', (b'Repository.insert_stream_1.19',))
        client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))
        client.add_expected_call(b'Repository.insert_stream', (b'quack/', b''), b'success', (b'ok',))

        class FakeRealSink:

            def __init__(self):
                self.records = []

            def insert_stream(self, stream, src_format, resume_tokens):
                for substream_kind, substream in stream:
                    self.records.append((substream_kind, [record.key for record in substream]))
                return ([b'fake tokens'], [b'fake missing keys'])
        fake_real_sink = FakeRealSink()

        class FakeRealRepository:

            def _get_sink(self):
                return fake_real_sink

            def is_in_write_group(self):
                return False

            def refresh_data(self):
                return True
        repo._real_repository = FakeRealRepository()
        sink = repo._get_sink()
        fmt = repository.format_registry.get_default()
        stream = self.make_stream_with_inv_deltas(fmt)
        resume_tokens, missing_keys = sink.insert_stream(stream, fmt, [])
        expected_records = [('inventory-deltas', [(b'rev2',), (b'rev3',)]), ('texts', [(b'some-rev', b'some-file')])]
        self.assertEqual(expected_records, fake_real_sink.records)
        self.assertEqual([b'fake tokens'], resume_tokens)
        self.assertEqual([b'fake missing keys'], missing_keys)
        self.assertFinished(client)

    def make_stream_with_inv_deltas(self, fmt):
        """Make a simple stream with an inventory delta followed by more
        records and more substreams to test that all records and substreams
        from that point on are used.

        This sends, in order:
           * inventories substream: rev1, rev2, rev3.  rev2 and rev3 are
             inventory-deltas.
           * texts substream: (some-rev, some-file)
        """
        inv = inventory.Inventory(revision_id=b'rev1')
        inv.root.revision = b'rev1'

        def stream_with_inv_delta():
            yield ('inventories', inventories_substream())
            yield ('inventory-deltas', inventory_delta_substream())
            yield ('texts', [versionedfile.FulltextContentFactory((b'some-rev', b'some-file'), (), None, b'content')])

        def inventories_substream():
            chunks = fmt._serializer.write_inventory_to_lines(inv)
            yield versionedfile.ChunkedContentFactory((b'rev1',), (), None, chunks, chunks_are_lines=True)

        def inventory_delta_substream():
            entry = inv.make_entry('directory', 'newdir', inv.root.file_id, b'newdir-id')
            entry.revision = b'ghost'
            delta = [(None, 'newdir', b'newdir-id', entry)]
            serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=False)
            lines = serializer.delta_to_lines(b'rev1', b'rev2', delta)
            yield versionedfile.ChunkedContentFactory((b'rev2',), (b'rev1',), None, lines)
            lines = serializer.delta_to_lines(b'rev1', b'rev3', delta)
            yield versionedfile.ChunkedContentFactory((b'rev3',), (b'rev1',), None, lines)
        return stream_with_inv_delta()