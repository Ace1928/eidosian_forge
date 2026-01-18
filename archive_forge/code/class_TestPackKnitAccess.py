import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
class TestPackKnitAccess(TestCaseWithMemoryTransport, KnitRecordAccessTestsMixin):
    """Tests for the pack based access."""

    def get_access(self):
        return self._get_access()[0]

    def _get_access(self, packname='packfile', index='FOO'):
        transport = self.get_transport()

        def write_data(bytes):
            transport.append_bytes(packname, bytes)
        writer = pack.ContainerWriter(write_data)
        writer.begin()
        access = pack_repo._DirectPackAccess({})
        access.set_writer(writer, index, (transport, packname))
        return (access, writer)

    def make_pack_file(self):
        """Create a pack file with 2 records."""
        access, writer = self._get_access(packname='packname', index='foo')
        memos = []
        memos.extend(access.add_raw_records([(b'key1', 10)], [b'1234567890']))
        memos.extend(access.add_raw_records([(b'key2', 5)], [b'12345']))
        writer.end()
        return memos

    def test_pack_collection_pack_retries(self):
        """An explicit pack of a pack collection succeeds even when a
        concurrent pack happens.
        """
        builder = self.make_branch_builder('.')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\nrev 1\n'))], revision_id=b'rev-1')
        builder.build_snapshot([b'rev-1'], [('modify', ('file', b'content\nrev 2\n'))], revision_id=b'rev-2')
        builder.build_snapshot([b'rev-2'], [('modify', ('file', b'content\nrev 3\n'))], revision_id=b'rev-3')
        self.addCleanup(builder.finish_series)
        b = builder.get_branch()
        self.addCleanup(b.lock_write().unlock)
        repo = b.repository
        collection = repo._pack_collection
        reopened_repo = repo.controldir.open_repository()
        reopened_repo.pack()
        collection.pack()

    def make_vf_for_retrying(self):
        """Create 3 packs and a reload function.

        Originally, 2 pack files will have the data, but one will be missing.
        And then the third will be used in place of the first two if reload()
        is called.

        :return: (versioned_file, reload_counter)
            versioned_file  a KnitVersionedFiles using the packs for access
        """
        builder = self.make_branch_builder('.', format='1.9')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'content\nrev 1\n'))], revision_id=b'rev-1')
        builder.build_snapshot([b'rev-1'], [('modify', ('file', b'content\nrev 2\n'))], revision_id=b'rev-2')
        builder.build_snapshot([b'rev-2'], [('modify', ('file', b'content\nrev 3\n'))], revision_id=b'rev-3')
        builder.finish_series()
        b = builder.get_branch()
        b.lock_write()
        self.addCleanup(b.unlock)
        repo = b.repository
        collection = repo._pack_collection
        collection.ensure_loaded()
        orig_packs = collection.packs
        packer = knitpack_repo.KnitPacker(collection, orig_packs, '.testpack')
        new_pack = packer.pack()
        collection.reset()
        repo.refresh_data()
        vf = repo.revisions
        new_index = new_pack.revision_index
        access_tuple = new_pack.access_tuple()
        reload_counter = [0, 0, 0]

        def reload():
            reload_counter[0] += 1
            if reload_counter[1] > 0:
                reload_counter[2] += 1
                return False
            reload_counter[1] += 1
            vf._index._graph_index._indices[:] = [new_index]
            vf._access._indices.clear()
            vf._access._indices[new_index] = access_tuple
            return True
        trans, name = orig_packs[1].access_tuple()
        trans.delete(name)
        vf._access._reload_func = reload
        return (vf, reload_counter)

    def make_reload_func(self, return_val=True):
        reload_called = [0]

        def reload():
            reload_called[0] += 1
            return return_val
        return (reload_called, reload)

    def make_retry_exception(self):
        try:
            raise _TestException('foobar')
        except _TestException as e:
            retry_exc = pack_repo.RetryWithNewPacks(None, reload_occurred=False, exc_info=sys.exc_info())
        return retry_exc

    def test_read_from_several_packs(self):
        access, writer = self._get_access()
        memos = []
        memos.extend(access.add_raw_records([(b'key', 10)], [b'1234567890']))
        writer.end()
        access, writer = self._get_access('pack2', 'FOOBAR')
        memos.extend(access.add_raw_records([(b'key', 5)], [b'12345']))
        writer.end()
        access, writer = self._get_access('pack3', 'BAZ')
        memos.extend(access.add_raw_records([(b'key', 5)], [b'alpha']))
        writer.end()
        transport = self.get_transport()
        access = pack_repo._DirectPackAccess({'FOO': (transport, 'packfile'), 'FOOBAR': (transport, 'pack2'), 'BAZ': (transport, 'pack3')})
        self.assertEqual([b'1234567890', b'12345', b'alpha'], list(access.get_raw_records(memos)))
        self.assertEqual([b'1234567890'], list(access.get_raw_records(memos[0:1])))
        self.assertEqual([b'12345'], list(access.get_raw_records(memos[1:2])))
        self.assertEqual([b'alpha'], list(access.get_raw_records(memos[2:3])))
        self.assertEqual([b'1234567890', b'alpha'], list(access.get_raw_records(memos[0:1] + memos[2:3])))

    def test_set_writer(self):
        """The writer should be settable post construction."""
        access = pack_repo._DirectPackAccess({})
        transport = self.get_transport()
        packname = 'packfile'
        index = 'foo'

        def write_data(bytes):
            transport.append_bytes(packname, bytes)
        writer = pack.ContainerWriter(write_data)
        writer.begin()
        access.set_writer(writer, index, (transport, packname))
        memos = access.add_raw_records([(b'key', 10)], [b'1234567890'])
        writer.end()
        self.assertEqual([b'1234567890'], list(access.get_raw_records(memos)))

    def test_missing_index_raises_retry(self):
        memos = self.make_pack_file()
        transport = self.get_transport()
        reload_called, reload_func = self.make_reload_func()
        access = pack_repo._DirectPackAccess({'bar': (transport, 'packname')}, reload_func=reload_func)
        e = self.assertListRaises(pack_repo.RetryWithNewPacks, access.get_raw_records, memos)
        self.assertTrue(e.reload_occurred)
        self.assertIsInstance(e.exc_info, tuple)
        self.assertIs(e.exc_info[0], KeyError)
        self.assertIsInstance(e.exc_info[1], KeyError)

    def test_missing_index_raises_key_error_with_no_reload(self):
        memos = self.make_pack_file()
        transport = self.get_transport()
        access = pack_repo._DirectPackAccess({'bar': (transport, 'packname')})
        e = self.assertListRaises(KeyError, access.get_raw_records, memos)

    def test_missing_file_raises_retry(self):
        memos = self.make_pack_file()
        transport = self.get_transport()
        reload_called, reload_func = self.make_reload_func()
        access = pack_repo._DirectPackAccess({'foo': (transport, 'different-packname')}, reload_func=reload_func)
        e = self.assertListRaises(pack_repo.RetryWithNewPacks, access.get_raw_records, memos)
        self.assertFalse(e.reload_occurred)
        self.assertIsInstance(e.exc_info, tuple)
        self.assertIs(e.exc_info[0], _mod_transport.NoSuchFile)
        self.assertIsInstance(e.exc_info[1], _mod_transport.NoSuchFile)
        self.assertEqual('different-packname', e.exc_info[1].path)

    def test_missing_file_raises_no_such_file_with_no_reload(self):
        memos = self.make_pack_file()
        transport = self.get_transport()
        access = pack_repo._DirectPackAccess({'foo': (transport, 'different-packname')})
        e = self.assertListRaises(_mod_transport.NoSuchFile, access.get_raw_records, memos)

    def test_failing_readv_raises_retry(self):
        memos = self.make_pack_file()
        transport = self.get_transport()
        failing_transport = MockReadvFailingTransport([transport.get_bytes('packname')])
        reload_called, reload_func = self.make_reload_func()
        access = pack_repo._DirectPackAccess({'foo': (failing_transport, 'packname')}, reload_func=reload_func)
        self.assertEqual([b'1234567890'], list(access.get_raw_records(memos[:1])))
        self.assertEqual([b'12345'], list(access.get_raw_records(memos[1:2])))
        e = self.assertListRaises(pack_repo.RetryWithNewPacks, access.get_raw_records, memos)
        self.assertFalse(e.reload_occurred)
        self.assertIsInstance(e.exc_info, tuple)
        self.assertIs(e.exc_info[0], _mod_transport.NoSuchFile)
        self.assertIsInstance(e.exc_info[1], _mod_transport.NoSuchFile)
        self.assertEqual('packname', e.exc_info[1].path)

    def test_failing_readv_raises_no_such_file_with_no_reload(self):
        memos = self.make_pack_file()
        transport = self.get_transport()
        failing_transport = MockReadvFailingTransport([transport.get_bytes('packname')])
        reload_called, reload_func = self.make_reload_func()
        access = pack_repo._DirectPackAccess({'foo': (failing_transport, 'packname')})
        self.assertEqual([b'1234567890'], list(access.get_raw_records(memos[:1])))
        self.assertEqual([b'12345'], list(access.get_raw_records(memos[1:2])))
        e = self.assertListRaises(_mod_transport.NoSuchFile, access.get_raw_records, memos)

    def test_reload_or_raise_no_reload(self):
        access = pack_repo._DirectPackAccess({}, reload_func=None)
        retry_exc = self.make_retry_exception()
        self.assertRaises(_TestException, access.reload_or_raise, retry_exc)

    def test_reload_or_raise_reload_changed(self):
        reload_called, reload_func = self.make_reload_func(return_val=True)
        access = pack_repo._DirectPackAccess({}, reload_func=reload_func)
        retry_exc = self.make_retry_exception()
        access.reload_or_raise(retry_exc)
        self.assertEqual([1], reload_called)
        retry_exc.reload_occurred = True
        access.reload_or_raise(retry_exc)
        self.assertEqual([2], reload_called)

    def test_reload_or_raise_reload_no_change(self):
        reload_called, reload_func = self.make_reload_func(return_val=False)
        access = pack_repo._DirectPackAccess({}, reload_func=reload_func)
        retry_exc = self.make_retry_exception()
        self.assertRaises(_TestException, access.reload_or_raise, retry_exc)
        self.assertEqual([1], reload_called)
        retry_exc.reload_occurred = True
        access.reload_or_raise(retry_exc)
        self.assertEqual([2], reload_called)

    def test_annotate_retries(self):
        vf, reload_counter = self.make_vf_for_retrying()
        key = (b'rev-3',)
        reload_lines = vf.annotate(key)
        self.assertEqual([1, 1, 0], reload_counter)
        plain_lines = vf.annotate(key)
        self.assertEqual([1, 1, 0], reload_counter)
        if reload_lines != plain_lines:
            self.fail('Annotation was not identical with reloading.')
        for trans, name in vf._access._indices.values():
            trans.delete(name)
        self.assertRaises(_mod_transport.NoSuchFile, vf.annotate, key)
        self.assertEqual([2, 1, 1], reload_counter)

    def test__get_record_map_retries(self):
        vf, reload_counter = self.make_vf_for_retrying()
        keys = [(b'rev-1',), (b'rev-2',), (b'rev-3',)]
        records = vf._get_record_map(keys)
        self.assertEqual(keys, sorted(records.keys()))
        self.assertEqual([1, 1, 0], reload_counter)
        for trans, name in vf._access._indices.values():
            trans.delete(name)
        self.assertRaises(_mod_transport.NoSuchFile, vf._get_record_map, keys)
        self.assertEqual([2, 1, 1], reload_counter)

    def test_get_record_stream_retries(self):
        vf, reload_counter = self.make_vf_for_retrying()
        keys = [(b'rev-1',), (b'rev-2',), (b'rev-3',)]
        record_stream = vf.get_record_stream(keys, 'topological', False)
        record = next(record_stream)
        self.assertEqual((b'rev-1',), record.key)
        self.assertEqual([0, 0, 0], reload_counter)
        record = next(record_stream)
        self.assertEqual((b'rev-2',), record.key)
        self.assertEqual([1, 1, 0], reload_counter)
        record = next(record_stream)
        self.assertEqual((b'rev-3',), record.key)
        self.assertEqual([1, 1, 0], reload_counter)
        for trans, name in vf._access._indices.values():
            trans.delete(name)
        self.assertListRaises(_mod_transport.NoSuchFile, vf.get_record_stream, keys, 'topological', False)

    def test_iter_lines_added_or_present_in_keys_retries(self):
        vf, reload_counter = self.make_vf_for_retrying()
        keys = [(b'rev-1',), (b'rev-2',), (b'rev-3',)]
        count = 0
        reload_lines = sorted(vf.iter_lines_added_or_present_in_keys(keys))
        self.assertEqual([1, 1, 0], reload_counter)
        plain_lines = sorted(vf.iter_lines_added_or_present_in_keys(keys))
        self.assertEqual([1, 1, 0], reload_counter)
        self.assertEqual(plain_lines, reload_lines)
        self.assertEqual(21, len(plain_lines))
        for trans, name in vf._access._indices.values():
            trans.delete(name)
        self.assertListRaises(_mod_transport.NoSuchFile, vf.iter_lines_added_or_present_in_keys, keys)
        self.assertEqual([2, 1, 1], reload_counter)

    def test_get_record_stream_yields_disk_sorted_order(self):
        repo = self.make_repository('test', format='pack-0.92')
        repo.lock_write()
        self.addCleanup(repo.unlock)
        repo.start_write_group()
        vf = repo.texts
        vf.add_lines((b'f-id', b'rev-5'), [(b'f-id', b'rev-4')], [b'lines\n'])
        vf.add_lines((b'f-id', b'rev-1'), [], [b'lines\n'])
        vf.add_lines((b'f-id', b'rev-2'), [(b'f-id', b'rev-1')], [b'lines\n'])
        repo.commit_write_group()
        stream = vf.get_record_stream([(b'f-id', b'rev-1'), (b'f-id', b'rev-5'), (b'f-id', b'rev-2')], 'unordered', False)
        keys = [r.key for r in stream]
        self.assertEqual([(b'f-id', b'rev-5'), (b'f-id', b'rev-1'), (b'f-id', b'rev-2')], keys)
        repo.start_write_group()
        vf.add_lines((b'f-id', b'rev-4'), [(b'f-id', b'rev-3')], [b'lines\n'])
        vf.add_lines((b'f-id', b'rev-3'), [(b'f-id', b'rev-2')], [b'lines\n'])
        vf.add_lines((b'f-id', b'rev-6'), [(b'f-id', b'rev-5')], [b'lines\n'])
        repo.commit_write_group()
        request_keys = {(b'f-id', b'rev-%d' % i) for i in range(1, 7)}
        stream = vf.get_record_stream(request_keys, 'unordered', False)
        keys = [r.key for r in stream]
        alt1 = [(b'f-id', b'rev-%d' % i) for i in [4, 3, 6, 5, 1, 2]]
        alt2 = [(b'f-id', b'rev-%d' % i) for i in [5, 1, 2, 4, 3, 6]]
        if keys != alt1 and keys != alt2:
            self.fail('Returned key order did not match either expected order. expected %s or %s, not %s' % (alt1, alt2, keys))