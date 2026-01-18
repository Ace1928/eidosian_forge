import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestAllGroupCompressors(TestGroupCompressor):
    """Tests for GroupCompressor"""
    scenarios = group_compress_implementation_scenarios()
    compressor = None

    def test_empty_delta(self):
        compressor = self.compressor()
        self.assertEqual([], compressor.chunks)

    def test_one_nosha_delta(self):
        compressor = self.compressor()
        text = b'strange\ncommon\n'
        sha1, start_point, end_point, _ = compressor.compress(('label',), [text], len(text), None)
        self.assertEqual(sha_string(b'strange\ncommon\n'), sha1)
        expected_lines = b'f\x0fstrange\ncommon\n'
        self.assertEqual(expected_lines, b''.join(compressor.chunks))
        self.assertEqual(0, start_point)
        self.assertEqual(len(expected_lines), end_point)

    def test_empty_content(self):
        compressor = self.compressor()
        sha1, start_point, end_point, kind = compressor.compress(('empty',), [], 0, None)
        self.assertEqual(0, start_point)
        self.assertEqual(0, end_point)
        self.assertEqual('fulltext', kind)
        self.assertEqual(groupcompress._null_sha1, sha1)
        self.assertEqual(0, compressor.endpoint)
        self.assertEqual([], compressor.chunks)
        text = b'some\nbytes\n'
        compressor.compress(('content',), [text], len(text), None)
        self.assertTrue(compressor.endpoint > 0)
        sha1, start_point, end_point, kind = compressor.compress(('empty2',), [], 0, None)
        self.assertEqual(0, start_point)
        self.assertEqual(0, end_point)
        self.assertEqual('fulltext', kind)
        self.assertEqual(groupcompress._null_sha1, sha1)

    def test_extract_from_compressor(self):
        compressor = self.compressor()
        text = b'strange\ncommon long line\nthat needs a 16 byte match\n'
        sha1_1, _, _, _ = compressor.compress(('label',), [text], len(text), None)
        expected_lines = list(compressor.chunks)
        text = b'common long line\nthat needs a 16 byte match\ndifferent\n'
        sha1_2, _, end_point, _ = compressor.compress(('newlabel',), [text], len(text), None)
        self.assertEqual(([b'strange\ncommon long line\nthat needs a 16 byte match\n'], sha1_1), compressor.extract(('label',)))
        self.assertEqual(([b'common long line\nthat needs a 16 byte match\ndifferent\n'], sha1_2), compressor.extract(('newlabel',)))

    def test_pop_last(self):
        compressor = self.compressor()
        text = b'some text\nfor the first entry\n'
        _, _, _, _ = compressor.compress(('key1',), [text], len(text), None)
        expected_lines = list(compressor.chunks)
        text = b'some text\nfor the second entry\n'
        _, _, _, _ = compressor.compress(('key2',), [text], len(text), None)
        compressor.pop_last()
        self.assertEqual(expected_lines, compressor.chunks)