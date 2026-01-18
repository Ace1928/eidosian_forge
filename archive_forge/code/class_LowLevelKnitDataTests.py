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
class LowLevelKnitDataTests(TestCase):

    def create_gz_content(self, text):
        sio = BytesIO()
        with gzip.GzipFile(mode='wb', fileobj=sio) as gz_file:
            gz_file.write(text)
        return sio.getvalue()

    def make_multiple_records(self):
        """Create the content for multiple records."""
        sha1sum = osutils.sha_string(b'foo\nbar\n')
        total_txt = []
        gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
        record_1 = (0, len(gz_txt), sha1sum)
        total_txt.append(gz_txt)
        sha1sum = osutils.sha_string(b'baz\n')
        gz_txt = self.create_gz_content(b'version rev-id-2 1 %s\nbaz\nend rev-id-2\n' % (sha1sum,))
        record_2 = (record_1[1], len(gz_txt), sha1sum)
        total_txt.append(gz_txt)
        return (total_txt, record_1, record_2)

    def test_valid_knit_data(self):
        sha1sum = osutils.sha_string(b'foo\nbar\n')
        gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
        transport = MockTransport([gz_txt])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(gz_txt)))]
        contents = list(knit._read_records_iter(records))
        self.assertEqual([((b'rev-id-1',), [b'foo\n', b'bar\n'], b'4e48e2c9a3d2ca8a708cb0cc545700544efb5021')], contents)
        raw_contents = list(knit._read_records_iter_raw(records))
        self.assertEqual([((b'rev-id-1',), gz_txt, sha1sum)], raw_contents)

    def test_multiple_records_valid(self):
        total_txt, record_1, record_2 = self.make_multiple_records()
        transport = MockTransport([b''.join(total_txt)])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-1',), ((b'rev-id-1',), record_1[0], record_1[1])), ((b'rev-id-2',), ((b'rev-id-2',), record_2[0], record_2[1]))]
        contents = list(knit._read_records_iter(records))
        self.assertEqual([((b'rev-id-1',), [b'foo\n', b'bar\n'], record_1[2]), ((b'rev-id-2',), [b'baz\n'], record_2[2])], contents)
        raw_contents = list(knit._read_records_iter_raw(records))
        self.assertEqual([((b'rev-id-1',), total_txt[0], record_1[2]), ((b'rev-id-2',), total_txt[1], record_2[2])], raw_contents)

    def test_not_enough_lines(self):
        sha1sum = osutils.sha_string(b'foo\n')
        gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nend rev-id-1\n' % (sha1sum,))
        transport = MockTransport([gz_txt])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(gz_txt)))]
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter(records))
        raw_contents = list(knit._read_records_iter_raw(records))
        self.assertEqual([((b'rev-id-1',), gz_txt, sha1sum)], raw_contents)

    def test_too_many_lines(self):
        sha1sum = osutils.sha_string(b'foo\nbar\n')
        gz_txt = self.create_gz_content(b'version rev-id-1 1 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
        transport = MockTransport([gz_txt])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(gz_txt)))]
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter(records))
        raw_contents = list(knit._read_records_iter_raw(records))
        self.assertEqual([((b'rev-id-1',), gz_txt, sha1sum)], raw_contents)

    def test_mismatched_version_id(self):
        sha1sum = osutils.sha_string(b'foo\nbar\n')
        gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
        transport = MockTransport([gz_txt])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-2',), ((b'rev-id-2',), 0, len(gz_txt)))]
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter(records))
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter_raw(records))

    def test_uncompressed_data(self):
        sha1sum = osutils.sha_string(b'foo\nbar\n')
        txt = b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,)
        transport = MockTransport([txt])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(txt)))]
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter(records))
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter_raw(records))

    def test_corrupted_data(self):
        sha1sum = osutils.sha_string(b'foo\nbar\n')
        gz_txt = self.create_gz_content(b'version rev-id-1 2 %s\nfoo\nbar\nend rev-id-1\n' % (sha1sum,))
        gz_txt = gz_txt[:10] + b'\xff\xff' + gz_txt[12:]
        transport = MockTransport([gz_txt])
        access = _KnitKeyAccess(transport, ConstantMapper('filename'))
        knit = KnitVersionedFiles(None, access)
        records = [((b'rev-id-1',), ((b'rev-id-1',), 0, len(gz_txt)))]
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter(records))
        self.assertRaises(KnitCorrupt, list, knit._read_records_iter_raw(records))