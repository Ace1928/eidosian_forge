import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
class TestDeltaIndex(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.requireFeature(compiled_groupcompress_feature)
        self._gc_module = compiled_groupcompress_feature.module

    def test_repr(self):
        di = self._gc_module.DeltaIndex(b'test text\n')
        self.assertEqual('DeltaIndex(1, 10)', repr(di))

    def test_sizeof(self):
        di = self._gc_module.DeltaIndex()
        lower_bound = di._max_num_sources * 12
        self.assertGreater(sys.getsizeof(di), lower_bound)

    def test__dump_no_index(self):
        di = self._gc_module.DeltaIndex()
        self.assertEqual(None, di._dump_index())

    def test__dump_index_simple(self):
        di = self._gc_module.DeltaIndex()
        di.add_source(_text1, 0)
        self.assertFalse(di._has_index())
        self.assertEqual(None, di._dump_index())
        _ = di.make_delta(_text1)
        self.assertTrue(di._has_index())
        hash_list, entry_list = di._dump_index()
        self.assertEqual(16, len(hash_list))
        self.assertEqual(68, len(entry_list))
        just_entries = [(idx, text_offset, hash_val) for idx, (text_offset, hash_val) in enumerate(entry_list) if text_offset != 0 or hash_val != 0]
        rabin_hash = self._gc_module._rabin_hash
        self.assertEqual([(8, 16, rabin_hash(_text1[1:17])), (25, 48, rabin_hash(_text1[33:49])), (34, 32, rabin_hash(_text1[17:33])), (47, 64, rabin_hash(_text1[49:65]))], just_entries)
        for entry_idx, text_offset, hash_val in just_entries:
            self.assertEqual(entry_idx, hash_list[hash_val & 15])

    def test__dump_index_two_sources(self):
        di = self._gc_module.DeltaIndex()
        di.add_source(_text1, 0)
        di.add_source(_text2, 2)
        start2 = len(_text1) + 2
        self.assertTrue(di._has_index())
        hash_list, entry_list = di._dump_index()
        self.assertEqual(16, len(hash_list))
        self.assertEqual(68, len(entry_list))
        just_entries = [(idx, text_offset, hash_val) for idx, (text_offset, hash_val) in enumerate(entry_list) if text_offset != 0 or hash_val != 0]
        rabin_hash = self._gc_module._rabin_hash
        self.assertEqual([(8, 16, rabin_hash(_text1[1:17])), (9, start2 + 16, rabin_hash(_text2[1:17])), (25, 48, rabin_hash(_text1[33:49])), (30, start2 + 64, rabin_hash(_text2[49:65])), (34, 32, rabin_hash(_text1[17:33])), (35, start2 + 32, rabin_hash(_text2[17:33])), (43, start2 + 48, rabin_hash(_text2[33:49])), (47, 64, rabin_hash(_text1[49:65]))], just_entries)
        for entry_idx, text_offset, hash_val in just_entries:
            hash_idx = hash_val & 15
            self.assertTrue(hash_list[hash_idx] <= entry_idx < hash_list[hash_idx + 1])

    def test_first_add_source_doesnt_index_until_make_delta(self):
        di = self._gc_module.DeltaIndex()
        self.assertFalse(di._has_index())
        di.add_source(_text1, 0)
        self.assertFalse(di._has_index())
        delta = di.make_delta(_text2)
        self.assertTrue(di._has_index())
        self.assertEqual(b'N\x90/\x1fdiffer from\nagainst other text\n', delta)

    def test_add_source_max_bytes_to_index(self):
        di = self._gc_module.DeltaIndex()
        di._max_bytes_to_index = 3 * 16
        di.add_source(_text1, 0)
        di.add_source(_text3, 3)
        start2 = len(_text1) + 3
        hash_list, entry_list = di._dump_index()
        self.assertEqual(16, len(hash_list))
        self.assertEqual(67, len(entry_list))
        just_entries = sorted([(text_offset, hash_val) for text_offset, hash_val in entry_list if text_offset != 0 or hash_val != 0])
        rabin_hash = self._gc_module._rabin_hash
        self.assertEqual([(25, rabin_hash(_text1[10:26])), (50, rabin_hash(_text1[35:51])), (75, rabin_hash(_text1[60:76])), (start2 + 44, rabin_hash(_text3[29:45])), (start2 + 88, rabin_hash(_text3[73:89])), (start2 + 132, rabin_hash(_text3[117:133]))], just_entries)

    def test_second_add_source_triggers_make_index(self):
        di = self._gc_module.DeltaIndex()
        self.assertFalse(di._has_index())
        di.add_source(_text1, 0)
        self.assertFalse(di._has_index())
        di.add_source(_text2, 0)
        self.assertTrue(di._has_index())

    def test_make_delta(self):
        di = self._gc_module.DeltaIndex(_text1)
        delta = di.make_delta(_text2)
        self.assertEqual(b'N\x90/\x1fdiffer from\nagainst other text\n', delta)

    def test_delta_against_multiple_sources(self):
        di = self._gc_module.DeltaIndex()
        di.add_source(_first_text, 0)
        self.assertEqual(len(_first_text), di._source_offset)
        di.add_source(_second_text, 0)
        self.assertEqual(len(_first_text) + len(_second_text), di._source_offset)
        delta = di.make_delta(_third_text)
        result = self._gc_module.apply_delta(_first_text + _second_text, delta)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual(b'\x85\x01\x90\x14\x0chas some in \x91v6\x03and\x91d"\x91:\n', delta)

    def test_delta_with_offsets(self):
        di = self._gc_module.DeltaIndex()
        di.add_source(_first_text, 5)
        self.assertEqual(len(_first_text) + 5, di._source_offset)
        di.add_source(_second_text, 10)
        self.assertEqual(len(_first_text) + len(_second_text) + 15, di._source_offset)
        delta = di.make_delta(_third_text)
        self.assertIsNot(None, delta)
        result = self._gc_module.apply_delta(b'12345' + _first_text + b'1234567890' + _second_text, delta)
        self.assertIsNot(None, result)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual(b'\x85\x01\x91\x05\x14\x0chas some in \x91\x856\x03and\x91s"\x91?\n', delta)

    def test_delta_with_delta_bytes(self):
        di = self._gc_module.DeltaIndex()
        source = _first_text
        di.add_source(_first_text, 0)
        self.assertEqual(len(_first_text), di._source_offset)
        delta = di.make_delta(_second_text)
        self.assertEqual(b'h\tsome more\x91\x019&previous text\nand has some extra text\n', delta)
        di.add_delta_source(delta, 0)
        source += delta
        self.assertEqual(len(_first_text) + len(delta), di._source_offset)
        second_delta = di.make_delta(_third_text)
        result = self._gc_module.apply_delta(source, second_delta)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual(b'\x85\x01\x90\x14\x1chas some in common with the \x91S&\x03and\x91\x18,', second_delta)
        di.add_delta_source(second_delta, 0)
        source += second_delta
        third_delta = di.make_delta(_third_text)
        result = self._gc_module.apply_delta(source, third_delta)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual(b'\x85\x01\x90\x14\x91~\x1c\x91S&\x03and\x91\x18,', third_delta)
        fourth_delta = di.make_delta(_fourth_text)
        self.assertEqual(_fourth_text, self._gc_module.apply_delta(source, fourth_delta))
        self.assertEqual(b'\x80\x01\x7f123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\x01\n', fourth_delta)
        di.add_delta_source(fourth_delta, 0)
        source += fourth_delta
        fifth_delta = di.make_delta(_fourth_text)
        self.assertEqual(_fourth_text, self._gc_module.apply_delta(source, fifth_delta))
        self.assertEqual(b'\x80\x01\x91\xa7\x7f\x01\n', fifth_delta)