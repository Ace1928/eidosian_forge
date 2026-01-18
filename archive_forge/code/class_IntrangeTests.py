import unittest
from idna.intranges import intranges_from_list, intranges_contain, _encode_range
class IntrangeTests(unittest.TestCase):

    def test_ranging(self):
        self.assertEqual(intranges_from_list(list(range(293, 499)) + list(range(4888, 9876))), (_encode_range(293, 499), _encode_range(4888, 9876)))

    def test_ranging_2(self):
        self.assertEqual(intranges_from_list([111]), (_encode_range(111, 112),))

    def test_skips(self):
        self.assertEqual(intranges_from_list([0, 2, 4, 6, 9, 10, 11, 13, 15]), (_encode_range(0, 1), _encode_range(2, 3), _encode_range(4, 5), _encode_range(6, 7), _encode_range(9, 12), _encode_range(13, 14), _encode_range(15, 16)))

    def test_empty_range(self):
        self.assertEqual(intranges_from_list([]), ())