import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestPyrexGroupCompressor(TestGroupCompressor):
    _test_needs_features = [compiled_groupcompress_feature]
    compressor = groupcompress.PyrexGroupCompressor

    def test_stats(self):
        compressor = self.compressor()
        chunks = [b'strange\n', b'common very very long line\n', b'plus more text\n']
        compressor.compress(('label',), chunks, sum(map(len, chunks)), None)
        chunks = [b'common very very long line\n', b'plus more text\n', b'different\n', b'moredifferent\n']
        compressor.compress(('newlabel',), chunks, sum(map(len, chunks)), None)
        chunks = [b'new\n', b'common very very long line\n', b'plus more text\n', b'different\n', b'moredifferent\n']
        compressor.compress(('label3',), chunks, sum(map(len, chunks)), None)
        self.assertAlmostEqual(1.9, compressor.ratio(), 1)

    def test_two_nosha_delta(self):
        compressor = self.compressor()
        text = b'strange\ncommon long line\nthat needs a 16 byte match\n'
        sha1_1, _, _, _ = compressor.compress(('label',), [text], len(text), None)
        expected_lines = list(compressor.chunks)
        text = b'common long line\nthat needs a 16 byte match\ndifferent\n'
        sha1_2, start_point, end_point, _ = compressor.compress(('newlabel',), [text], len(text), None)
        self.assertEqual(sha_string(text), sha1_2)
        expected_lines.extend([b'd\x0f', b'6', b'\x91\n,', b'\ndifferent\n'])
        self.assertEqualDiffEncoded(expected_lines, compressor.chunks)
        self.assertEqual(sum(map(len, expected_lines)), end_point)

    def test_three_nosha_delta(self):
        compressor = self.compressor()
        text = b'strange\ncommon very very long line\nwith some extra text\n'
        sha1_1, _, _, _ = compressor.compress(('label',), [text], len(text), None)
        text = b'different\nmoredifferent\nand then some more\n'
        sha1_2, _, _, _ = compressor.compress(('newlabel',), [text], len(text), None)
        expected_lines = list(compressor.chunks)
        text = b'new\ncommon very very long line\nwith some extra text\ndifferent\nmoredifferent\nand then some more\n'
        sha1_3, start_point, end_point, _ = compressor.compress(('label3',), [text], len(text), None)
        self.assertEqual(sha_string(text), sha1_3)
        expected_lines.extend([b'd\x0b', b'_\x03new', b'\x91\t1\x91<+'])
        self.assertEqualDiffEncoded(expected_lines, compressor.chunks)
        self.assertEqual(sum(map(len, expected_lines)), end_point)