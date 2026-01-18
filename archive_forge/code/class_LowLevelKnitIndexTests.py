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
class LowLevelKnitIndexTests(TestCase):

    @property
    def _load_data(self):
        from .._knit_load_data_py import _load_data_py
        return _load_data_py

    def get_knit_index(self, transport, name, mode):
        mapper = ConstantMapper(name)
        self.overrideAttr(knit, '_load_data', self._load_data)

        def allow_writes():
            return 'w' in mode
        return _KndxIndex(transport, mapper, lambda: None, allow_writes, lambda: True)

    def test_create_file(self):
        transport = MockTransport()
        index = self.get_knit_index(transport, 'filename', 'w')
        index.keys()
        call = transport.calls.pop(0)
        self.assertEqual('put_file_non_atomic', call[0])
        self.assertEqual('filename.kndx', call[1][0])
        self.assertEqual(_KndxIndex.HEADER, call[1][1].getvalue())
        self.assertEqual({'create_parent_dir': True}, call[2])

    def test_read_utf8_version_id(self):
        unicode_revision_id = 'version-А'
        utf8_revision_id = unicode_revision_id.encode('utf-8')
        transport = MockTransport([_KndxIndex.HEADER, b'%s option 0 1 :' % (utf8_revision_id,)])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(utf8_revision_id,): ()}, index.get_parent_map(index.keys()))
        self.assertFalse((unicode_revision_id,) in index.keys())

    def test_read_utf8_parents(self):
        unicode_revision_id = 'version-А'
        utf8_revision_id = unicode_revision_id.encode('utf-8')
        transport = MockTransport([_KndxIndex.HEADER, b'version option 0 1 .%s :' % (utf8_revision_id,)])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(b'version',): ((utf8_revision_id,),)}, index.get_parent_map(index.keys()))

    def test_read_ignore_corrupted_lines(self):
        transport = MockTransport([_KndxIndex.HEADER, b'corrupted', b'corrupted options 0 1 .b .c ', b'version options 0 1 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual(1, len(index.keys()))
        self.assertEqual({(b'version',)}, index.keys())

    def test_read_corrupted_header(self):
        transport = MockTransport([b'not a bzr knit index header\n'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(KnitHeaderError, index.keys)

    def test_read_duplicate_entries(self):
        transport = MockTransport([_KndxIndex.HEADER, b'parent options 0 1 :', b'version options1 0 1 0 :', b'version options2 1 2 .other :', b'version options3 3 4 0 .other :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual(2, len(index.keys()))
        self.assertEqual(b'1', index._dictionary_compress([(b'version',)]))
        self.assertEqual(((b'version',), 3, 4), index.get_position((b'version',)))
        self.assertEqual([b'options3'], index.get_options((b'version',)))
        self.assertEqual({(b'version',): ((b'parent',), (b'other',))}, index.get_parent_map([(b'version',)]))

    def test_read_compressed_parents(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 0 1 0 :', b'c option 0 1 1 0 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(b'b',): ((b'a',),), (b'c',): ((b'b',), (b'a',))}, index.get_parent_map([(b'b',), (b'c',)]))

    def test_write_utf8_version_id(self):
        unicode_revision_id = 'version-А'
        utf8_revision_id = unicode_revision_id.encode('utf-8')
        transport = MockTransport([_KndxIndex.HEADER])
        index = self.get_knit_index(transport, 'filename', 'r')
        index.add_records([((utf8_revision_id,), [b'option'], ((utf8_revision_id,), 0, 1), [])])
        call = transport.calls.pop(0)
        self.assertEqual('put_file_non_atomic', call[0])
        self.assertEqual('filename.kndx', call[1][0])
        self.assertEqual(_KndxIndex.HEADER + b'\n%s option 0 1  :' % (utf8_revision_id,), call[1][1].getvalue())
        self.assertEqual({'create_parent_dir': True}, call[2])

    def test_write_utf8_parents(self):
        unicode_revision_id = 'version-А'
        utf8_revision_id = unicode_revision_id.encode('utf-8')
        transport = MockTransport([_KndxIndex.HEADER])
        index = self.get_knit_index(transport, 'filename', 'r')
        index.add_records([((b'version',), [b'option'], ((b'version',), 0, 1), [(utf8_revision_id,)])])
        call = transport.calls.pop(0)
        self.assertEqual('put_file_non_atomic', call[0])
        self.assertEqual('filename.kndx', call[1][0])
        self.assertEqual(_KndxIndex.HEADER + b'\nversion option 0 1 .%s :' % (utf8_revision_id,), call[1][1].getvalue())
        self.assertEqual({'create_parent_dir': True}, call[2])

    def test_keys(self):
        transport = MockTransport([_KndxIndex.HEADER])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual(set(), index.keys())
        index.add_records([((b'a',), [b'option'], ((b'a',), 0, 1), [])])
        self.assertEqual({(b'a',)}, index.keys())
        index.add_records([((b'a',), [b'option'], ((b'a',), 0, 1), [])])
        self.assertEqual({(b'a',)}, index.keys())
        index.add_records([((b'b',), [b'option'], ((b'b',), 0, 1), [])])
        self.assertEqual({(b'a',), (b'b',)}, index.keys())

    def add_a_b(self, index, random_id=None):
        kwargs = {}
        if random_id is not None:
            kwargs['random_id'] = random_id
        index.add_records([((b'a',), [b'option'], ((b'a',), 0, 1), [(b'b',)]), ((b'a',), [b'opt'], ((b'a',), 1, 2), [(b'c',)]), ((b'b',), [b'option'], ((b'b',), 2, 3), [(b'a',)])], **kwargs)

    def assertIndexIsAB(self, index):
        self.assertEqual({(b'a',): ((b'c',),), (b'b',): ((b'a',),)}, index.get_parent_map(index.keys()))
        self.assertEqual(((b'a',), 1, 2), index.get_position((b'a',)))
        self.assertEqual(((b'b',), 2, 3), index.get_position((b'b',)))
        self.assertEqual([b'opt'], index.get_options((b'a',)))

    def test_add_versions(self):
        transport = MockTransport([_KndxIndex.HEADER])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.add_a_b(index)
        call = transport.calls.pop(0)
        self.assertEqual('put_file_non_atomic', call[0])
        self.assertEqual('filename.kndx', call[1][0])
        self.assertEqual(_KndxIndex.HEADER + b'\na option 0 1 .b :\na opt 1 2 .c :\nb option 2 3 0 :', call[1][1].getvalue())
        self.assertEqual({'create_parent_dir': True}, call[2])
        self.assertIndexIsAB(index)

    def test_add_versions_random_id_is_accepted(self):
        transport = MockTransport([_KndxIndex.HEADER])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.add_a_b(index, random_id=True)

    def test_delay_create_and_add_versions(self):
        transport = MockTransport()
        index = self.get_knit_index(transport, 'filename', 'w')
        self.assertEqual([], transport.calls)
        self.add_a_b(index)
        self.assertEqual(2, len(transport.calls))
        call = transport.calls.pop(0)
        self.assertEqual('put_file_non_atomic', call[0])
        self.assertEqual('filename.kndx', call[1][0])
        self.assertEqual(_KndxIndex.HEADER, call[1][1].getvalue())
        self.assertEqual({'create_parent_dir': True}, call[2])
        call = transport.calls.pop(0)
        self.assertEqual('put_file_non_atomic', call[0])
        self.assertEqual('filename.kndx', call[1][0])
        self.assertEqual(_KndxIndex.HEADER + b'\na option 0 1 .b :\na opt 1 2 .c :\nb option 2 3 0 :', call[1][1].getvalue())
        self.assertEqual({'create_parent_dir': True}, call[2])

    def assertTotalBuildSize(self, size, keys, positions):
        self.assertEqual(size, knit._get_total_build_size(None, keys, positions))

    def test__get_total_build_size(self):
        positions = {(b'a',): (('fulltext', False), ((b'a',), 0, 100), None), (b'b',): (('line-delta', False), ((b'b',), 100, 21), (b'a',)), (b'c',): (('line-delta', False), ((b'c',), 121, 35), (b'b',)), (b'd',): (('line-delta', False), ((b'd',), 156, 12), (b'b',))}
        self.assertTotalBuildSize(100, [(b'a',)], positions)
        self.assertTotalBuildSize(121, [(b'b',)], positions)
        self.assertTotalBuildSize(156, [(b'c',)], positions)
        self.assertTotalBuildSize(156, [(b'b',), (b'c',)], positions)
        self.assertTotalBuildSize(133, [(b'd',)], positions)
        self.assertTotalBuildSize(168, [(b'c',), (b'd',)], positions)

    def test_get_position(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 1 2 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual(((b'a',), 0, 1), index.get_position((b'a',)))
        self.assertEqual(((b'b',), 1, 2), index.get_position((b'b',)))

    def test_get_method(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a fulltext,unknown 0 1 :', b'b unknown,line-delta 1 2 :', b'c bad 3 4 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual('fulltext', index.get_method(b'a'))
        self.assertEqual('line-delta', index.get_method(b'b'))
        self.assertRaises(knit.KnitIndexUnknownMethod, index.get_method, b'c')

    def test_get_options(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a opt1 0 1 :', b'b opt2,opt3 1 2 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual([b'opt1'], index.get_options(b'a'))
        self.assertEqual([b'opt2', b'opt3'], index.get_options(b'b'))

    def test_get_parent_map(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 1 2 0 .c :', b'c option 1 2 1 0 .e :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(b'a',): (), (b'b',): ((b'a',), (b'c',)), (b'c',): ((b'b',), (b'a',), (b'e',))}, index.get_parent_map(index.keys()))

    def test_impossible_parent(self):
        """Test we get KnitCorrupt if the parent couldn't possibly exist."""
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 0 1 4 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(KnitCorrupt, index.keys)

    def test_corrupted_parent(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 0 1 :', b'c option 0 1 1v :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(KnitCorrupt, index.keys)

    def test_corrupted_parent_in_list(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 1 :', b'b option 0 1 :', b'c option 0 1 1 v :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(KnitCorrupt, index.keys)

    def test_invalid_position(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 1v 1 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(KnitCorrupt, index.keys)

    def test_invalid_size(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 1 1v :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(KnitCorrupt, index.keys)

    def test_scan_unvalidated_index_not_implemented(self):
        transport = MockTransport()
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertRaises(NotImplementedError, index.scan_unvalidated_index, 'dummy graph_index')
        self.assertRaises(NotImplementedError, index.get_missing_compression_parents)

    def test_short_line(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 10  :', b'b option 10 10 0'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(b'a',)}, index.keys())

    def test_skip_incomplete_record(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 10  :', b'b option 10 10 0', b'c option 20 10 0 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(b'a',), (b'c',)}, index.keys())

    def test_trailing_characters(self):
        transport = MockTransport([_KndxIndex.HEADER, b'a option 0 10  :', b'b option 10 10 0 :a', b'c option 20 10 0 :'])
        index = self.get_knit_index(transport, 'filename', 'r')
        self.assertEqual({(b'a',), (b'c',)}, index.keys())