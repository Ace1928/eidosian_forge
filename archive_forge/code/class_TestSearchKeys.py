from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
class TestSearchKeys(tests.TestCase):
    module = None

    def assertSearchKey16(self, expected, key):
        self.assertEqual(expected, self.module._search_key_16(key))

    def assertSearchKey255(self, expected, key):
        actual = self.module._search_key_255(key)
        self.assertEqual(expected, actual, 'actual: {!r}'.format(actual))

    def test_simple_16(self):
        self.assertSearchKey16(b'8C736521', stuple(b'foo'))
        self.assertSearchKey16(b'8C736521\x008C736521', stuple(b'foo', b'foo'))
        self.assertSearchKey16(b'8C736521\x0076FF8CAA', stuple(b'foo', b'bar'))
        self.assertSearchKey16(b'ED82CD11', stuple(b'abcd'))

    def test_simple_255(self):
        self.assertSearchKey255(b'\x8cse!', stuple(b'foo'))
        self.assertSearchKey255(b'\x8cse!\x00\x8cse!', stuple(b'foo', b'foo'))
        self.assertSearchKey255(b'\x8cse!\x00v\xff\x8c\xaa', stuple(b'foo', b'bar'))
        self.assertSearchKey255(b'\xfdm\x93_\x00P_\x1bL', stuple(b'<', b'V'))

    def test_255_does_not_include_newline(self):
        chars_used = set()
        for char_in in range(256):
            search_key = self.module._search_key_255(stuple(bytes([char_in])))
            chars_used.update([bytes([x]) for x in search_key])
        all_chars = {bytes([x]) for x in range(256)}
        unused_chars = all_chars.symmetric_difference(chars_used)
        self.assertEqual({b'\n'}, unused_chars)