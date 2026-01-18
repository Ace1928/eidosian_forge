import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
class TestPatienceDiffLib(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._unique_lcs = _patiencediff_py.unique_lcs_py
        self._recurse_matches = _patiencediff_py.recurse_matches_py
        self._PatienceSequenceMatcher = _patiencediff_py.PatienceSequenceMatcher_py

    def test_diff_unicode_string(self):
        a = ''.join([chr(i) for i in range(4000, 4500, 3)])
        b = ''.join([chr(i) for i in range(4300, 4800, 2)])
        sm = self._PatienceSequenceMatcher(None, a, b)
        mb = sm.get_matching_blocks()
        self.assertEqual(35, len(mb))

    def test_unique_lcs(self):
        unique_lcs = self._unique_lcs
        self.assertEqual(unique_lcs('', ''), [])
        self.assertEqual(unique_lcs('', 'a'), [])
        self.assertEqual(unique_lcs('a', ''), [])
        self.assertEqual(unique_lcs('a', 'a'), [(0, 0)])
        self.assertEqual(unique_lcs('a', 'b'), [])
        self.assertEqual(unique_lcs('ab', 'ab'), [(0, 0), (1, 1)])
        self.assertEqual(unique_lcs('abcde', 'cdeab'), [(2, 0), (3, 1), (4, 2)])
        self.assertEqual(unique_lcs('cdeab', 'abcde'), [(0, 2), (1, 3), (2, 4)])
        self.assertEqual(unique_lcs('abXde', 'abYde'), [(0, 0), (1, 1), (3, 3), (4, 4)])
        self.assertEqual(unique_lcs('acbac', 'abc'), [(2, 1)])

    def test_recurse_matches(self):

        def test_one(a, b, matches):
            test_matches = []
            self._recurse_matches(a, b, 0, 0, len(a), len(b), test_matches, 10)
            self.assertEqual(test_matches, matches)
        test_one(['a', '', 'b', '', 'c'], ['a', 'a', 'b', 'c', 'c'], [(0, 0), (2, 2), (4, 4)])
        test_one(['a', 'c', 'b', 'a', 'c'], ['a', 'b', 'c'], [(0, 0), (2, 1), (4, 2)])
        test_one('abcdbce', 'afbcgdbce', [(0, 0), (1, 2), (2, 3), (3, 5), (4, 6), (5, 7), (6, 8)])
        test_one('aBccDe', 'abccde', [(0, 0), (5, 5)])

    def assertDiffBlocks(self, a, b, expected_blocks):
        """Check that the sequence matcher returns the correct blocks.

        :param a: A sequence to match
        :param b: Another sequence to match
        :param expected_blocks: The expected output, not including the final
            matching block (len(a), len(b), 0)
        """
        matcher = self._PatienceSequenceMatcher(None, a, b)
        blocks = matcher.get_matching_blocks()
        last = blocks.pop()
        self.assertEqual((len(a), len(b), 0), last)
        self.assertEqual(expected_blocks, blocks)

    def test_matching_blocks(self):
        self.assertDiffBlocks('', '', [])
        self.assertDiffBlocks([], [], [])
        self.assertDiffBlocks('abc', '', [])
        self.assertDiffBlocks('', 'abc', [])
        self.assertDiffBlocks('abcd', 'abcd', [(0, 0, 4)])
        self.assertDiffBlocks('abcd', 'abce', [(0, 0, 3)])
        self.assertDiffBlocks('eabc', 'abce', [(1, 0, 3)])
        self.assertDiffBlocks('eabce', 'abce', [(1, 0, 4)])
        self.assertDiffBlocks('abcde', 'abXde', [(0, 0, 2), (3, 3, 2)])
        self.assertDiffBlocks('abcde', 'abXYZde', [(0, 0, 2), (3, 5, 2)])
        self.assertDiffBlocks('abde', 'abXYZde', [(0, 0, 2), (2, 5, 2)])
        self.assertDiffBlocks('abcdefghijklmnop', 'abcdefxydefghijklmnop', [(0, 0, 6), (6, 11, 10)])
        self.assertDiffBlocks(['hello there\n', 'world\n', 'how are you today?\n'], ['hello there\n', 'how are you today?\n'], [(0, 0, 1), (2, 1, 1)])
        self.assertDiffBlocks('aBccDe', 'abccde', [(0, 0, 1), (5, 5, 1)])
        self.assertDiffBlocks('aBcDec', 'abcdec', [(0, 0, 1), (2, 2, 1), (4, 4, 2)])
        self.assertDiffBlocks('aBcdEcdFg', 'abcdecdfg', [(0, 0, 1), (8, 8, 1)])
        self.assertDiffBlocks('aBcdEeXcdFg', 'abcdecdfg', [(0, 0, 1), (2, 2, 2), (5, 4, 1), (7, 5, 2), (10, 8, 1)])
        self.assertDiffBlocks('abbabbXd', 'cabbabxd', [(7, 7, 1)])
        self.assertDiffBlocks('abbabbbb', 'cabbabbc', [])
        self.assertDiffBlocks('bbbbbbbb', 'cbbbbbbc', [])

    def test_matching_blocks_tuples(self):
        self.assertDiffBlocks([], [], [])
        self.assertDiffBlocks([('a',), ('b',), 'c,'], [], [])
        self.assertDiffBlocks([], [('a',), ('b',), 'c,'], [])
        self.assertDiffBlocks([('a',), ('b',), 'c,'], [('a',), ('b',), 'c,'], [(0, 0, 3)])
        self.assertDiffBlocks([('a',), ('b',), 'c,'], [('a',), ('b',), 'd,'], [(0, 0, 2)])
        self.assertDiffBlocks([('d',), ('b',), 'c,'], [('a',), ('b',), 'c,'], [(1, 1, 2)])
        self.assertDiffBlocks([('d',), ('a',), ('b',), 'c,'], [('a',), ('b',), 'c,'], [(1, 0, 3)])
        self.assertDiffBlocks([('a', 'b'), ('c', 'd'), ('e', 'f')], [('a', 'b'), ('c', 'X'), ('e', 'f')], [(0, 0, 1), (2, 2, 1)])
        self.assertDiffBlocks([('a', 'b'), ('c', 'd'), ('e', 'f')], [('a', 'b'), ('c', 'dX'), ('e', 'f')], [(0, 0, 1), (2, 2, 1)])

    def test_opcodes(self):

        def chk_ops(a, b, expected_codes):
            s = self._PatienceSequenceMatcher(None, a, b)
            self.assertEqual(expected_codes, s.get_opcodes())
        chk_ops('', '', [])
        chk_ops([], [], [])
        chk_ops('abc', '', [('delete', 0, 3, 0, 0)])
        chk_ops('', 'abc', [('insert', 0, 0, 0, 3)])
        chk_ops('abcd', 'abcd', [('equal', 0, 4, 0, 4)])
        chk_ops('abcd', 'abce', [('equal', 0, 3, 0, 3), ('replace', 3, 4, 3, 4)])
        chk_ops('eabc', 'abce', [('delete', 0, 1, 0, 0), ('equal', 1, 4, 0, 3), ('insert', 4, 4, 3, 4)])
        chk_ops('eabce', 'abce', [('delete', 0, 1, 0, 0), ('equal', 1, 5, 0, 4)])
        chk_ops('abcde', 'abXde', [('equal', 0, 2, 0, 2), ('replace', 2, 3, 2, 3), ('equal', 3, 5, 3, 5)])
        chk_ops('abcde', 'abXYZde', [('equal', 0, 2, 0, 2), ('replace', 2, 3, 2, 5), ('equal', 3, 5, 5, 7)])
        chk_ops('abde', 'abXYZde', [('equal', 0, 2, 0, 2), ('insert', 2, 2, 2, 5), ('equal', 2, 4, 5, 7)])
        chk_ops('abcdefghijklmnop', 'abcdefxydefghijklmnop', [('equal', 0, 6, 0, 6), ('insert', 6, 6, 6, 11), ('equal', 6, 16, 11, 21)])
        chk_ops(['hello there\n', 'world\n', 'how are you today?\n'], ['hello there\n', 'how are you today?\n'], [('equal', 0, 1, 0, 1), ('delete', 1, 2, 1, 1), ('equal', 2, 3, 1, 2)])
        chk_ops('aBccDe', 'abccde', [('equal', 0, 1, 0, 1), ('replace', 1, 5, 1, 5), ('equal', 5, 6, 5, 6)])
        chk_ops('aBcDec', 'abcdec', [('equal', 0, 1, 0, 1), ('replace', 1, 2, 1, 2), ('equal', 2, 3, 2, 3), ('replace', 3, 4, 3, 4), ('equal', 4, 6, 4, 6)])
        chk_ops('aBcdEcdFg', 'abcdecdfg', [('equal', 0, 1, 0, 1), ('replace', 1, 8, 1, 8), ('equal', 8, 9, 8, 9)])
        chk_ops('aBcdEeXcdFg', 'abcdecdfg', [('equal', 0, 1, 0, 1), ('replace', 1, 2, 1, 2), ('equal', 2, 4, 2, 4), ('delete', 4, 5, 4, 4), ('equal', 5, 6, 4, 5), ('delete', 6, 7, 5, 5), ('equal', 7, 9, 5, 7), ('replace', 9, 10, 7, 8), ('equal', 10, 11, 8, 9)])

    def test_grouped_opcodes(self):

        def chk_ops(a, b, expected_codes, n=3):
            s = self._PatienceSequenceMatcher(None, a, b)
            self.assertEqual(expected_codes, list(s.get_grouped_opcodes(n)))
        chk_ops('', '', [])
        chk_ops([], [], [])
        chk_ops('abc', '', [[('delete', 0, 3, 0, 0)]])
        chk_ops('', 'abc', [[('insert', 0, 0, 0, 3)]])
        chk_ops('abcd', 'abcd', [])
        chk_ops('abcd', 'abce', [[('equal', 0, 3, 0, 3), ('replace', 3, 4, 3, 4)]])
        chk_ops('eabc', 'abce', [[('delete', 0, 1, 0, 0), ('equal', 1, 4, 0, 3), ('insert', 4, 4, 3, 4)]])
        chk_ops('abcdefghijklmnop', 'abcdefxydefghijklmnop', [[('equal', 3, 6, 3, 6), ('insert', 6, 6, 6, 11), ('equal', 6, 9, 11, 14)]])
        chk_ops('abcdefghijklmnop', 'abcdefxydefghijklmnop', [[('equal', 2, 6, 2, 6), ('insert', 6, 6, 6, 11), ('equal', 6, 10, 11, 15)]], 4)
        chk_ops('Xabcdef', 'abcdef', [[('delete', 0, 1, 0, 0), ('equal', 1, 4, 0, 3)]])
        chk_ops('abcdef', 'abcdefX', [[('equal', 3, 6, 3, 6), ('insert', 6, 6, 6, 7)]])

    def test_multiple_ranges(self):
        self.assertDiffBlocks('abcdefghijklmnop', 'abcXghiYZQRSTUVWXYZijklmnop', [(0, 0, 3), (6, 4, 3), (9, 20, 7)])
        self.assertDiffBlocks('ABCd efghIjk  L', 'AxyzBCn mo pqrstuvwI1 2  L', [(0, 0, 1), (1, 4, 2), (9, 19, 1), (12, 23, 3)])
        self.assertDiffBlocks('    trg nqqrq jura lbh nqq n svyr va gur qverpgbel.\n    """\n    gnxrf_netf = [\'svyr*\']\n    gnxrf_bcgvbaf = [\'ab-erphefr\']\n\n    qrs eha(frys, svyr_yvfg, ab_erphefr=Snyfr):\n        sebz omeyvo.nqq vzcbeg fzneg_nqq, nqq_ercbegre_cevag, nqq_ercbegre_ahyy\n        vs vf_dhvrg():\n            ercbegre = nqq_ercbegre_ahyy\n        ryfr:\n            ercbegre = nqq_ercbegre_cevag\n        fzneg_nqq(svyr_yvfg, abg ab_erphefr, ercbegre)\n\n\npynff pzq_zxqve(Pbzznaq):\n'.splitlines(True), '    trg nqqrq jura lbh nqq n svyr va gur qverpgbel.\n\n    --qel-eha jvyy fubj juvpu svyrf jbhyq or nqqrq, ohg abg npghnyyl\n    nqq gurz.\n    """\n    gnxrf_netf = [\'svyr*\']\n    gnxrf_bcgvbaf = [\'ab-erphefr\', \'qel-eha\']\n\n    qrs eha(frys, svyr_yvfg, ab_erphefr=Snyfr, qel_eha=Snyfr):\n        vzcbeg omeyvo.nqq\n\n        vs qel_eha:\n            vs vf_dhvrg():\n                # Guvf vf cbvagyrff, ohg V\'q engure abg envfr na reebe\n                npgvba = omeyvo.nqq.nqq_npgvba_ahyy\n            ryfr:\n  npgvba = omeyvo.nqq.nqq_npgvba_cevag\n        ryvs vf_dhvrg():\n            npgvba = omeyvo.nqq.nqq_npgvba_nqq\n        ryfr:\n       npgvba = omeyvo.nqq.nqq_npgvba_nqq_naq_cevag\n\n        omeyvo.nqq.fzneg_nqq(svyr_yvfg, abg ab_erphefr, npgvba)\n\n\npynff pzq_zxqve(Pbzznaq):\n'.splitlines(True), [(0, 0, 1), (1, 4, 2), (9, 19, 1), (12, 23, 3)])

    def test_patience_unified_diff(self):
        txt_a = ['hello there\n', 'world\n', 'how are you today?\n']
        txt_b = ['hello there\n', 'how are you today?\n']
        unified_diff = patiencediff.unified_diff
        psm = self._PatienceSequenceMatcher
        self.assertEqual(['--- \n', '+++ \n', '@@ -1,3 +1,2 @@\n', ' hello there\n', '-world\n', ' how are you today?\n'], list(unified_diff(txt_a, txt_b, sequencematcher=psm)))
        txt_a = [x + '\n' for x in 'abcdefghijklmnop']
        txt_b = [x + '\n' for x in 'abcdefxydefghijklmnop']
        self.assertEqual(['--- \n', '+++ \n', '@@ -1,6 +1,11 @@\n', ' a\n', ' b\n', ' c\n', '+d\n', '+e\n', '+f\n', '+x\n', '+y\n', ' d\n', ' e\n', ' f\n'], list(unified_diff(txt_a, txt_b)))
        self.assertEqual(['--- \n', '+++ \n', '@@ -4,6 +4,11 @@\n', ' d\n', ' e\n', ' f\n', '+x\n', '+y\n', '+d\n', '+e\n', '+f\n', ' g\n', ' h\n', ' i\n'], list(unified_diff(txt_a, txt_b, sequencematcher=psm)))

    def test_patience_unified_diff_with_dates(self):
        txt_a = ['hello there\n', 'world\n', 'how are you today?\n']
        txt_b = ['hello there\n', 'how are you today?\n']
        unified_diff = patiencediff.unified_diff
        psm = self._PatienceSequenceMatcher
        self.assertEqual(['--- a\t2008-08-08\n', '+++ b\t2008-09-09\n', '@@ -1,3 +1,2 @@\n', ' hello there\n', '-world\n', ' how are you today?\n'], list(unified_diff(txt_a, txt_b, fromfile='a', tofile='b', fromfiledate='2008-08-08', tofiledate='2008-09-09', sequencematcher=psm)))