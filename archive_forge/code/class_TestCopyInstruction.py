import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
class TestCopyInstruction(tests.TestCase):

    def assertEncode(self, expected, offset, length):
        data = _groupcompress_py.encode_copy_instruction(offset, length)
        self.assertEqual(expected, data)

    def assertDecode(self, exp_offset, exp_length, exp_newpos, data, pos):
        cmd = data[pos]
        pos += 1
        out = _groupcompress_py.decode_copy_instruction(data, cmd, pos)
        self.assertEqual((exp_offset, exp_length, exp_newpos), out)

    def test_encode_no_length(self):
        self.assertEncode(b'\x80', 0, 64 * 1024)
        self.assertEncode(b'\x81\x01', 1, 64 * 1024)
        self.assertEncode(b'\x81\n', 10, 64 * 1024)
        self.assertEncode(b'\x81\xff', 255, 64 * 1024)
        self.assertEncode(b'\x82\x01', 256, 64 * 1024)
        self.assertEncode(b'\x83\x01\x01', 257, 64 * 1024)
        self.assertEncode(b'\x8f\xff\xff\xff\xff', 4294967295, 64 * 1024)
        self.assertEncode(b'\x8e\xff\xff\xff', 4294967040, 64 * 1024)
        self.assertEncode(b'\x8d\xff\xff\xff', 4294902015, 64 * 1024)
        self.assertEncode(b'\x8b\xff\xff\xff', 4278255615, 64 * 1024)
        self.assertEncode(b'\x87\xff\xff\xff', 16777215, 64 * 1024)
        self.assertEncode(b'\x8f\x04\x03\x02\x01', 16909060, 64 * 1024)

    def test_encode_no_offset(self):
        self.assertEncode(b'\x90\x01', 0, 1)
        self.assertEncode(b'\x90\n', 0, 10)
        self.assertEncode(b'\x90\xff', 0, 255)
        self.assertEncode(b'\xa0\x01', 0, 256)
        self.assertEncode(b'\xb0\x01\x01', 0, 257)
        self.assertEncode(b'\xb0\xff\xff', 0, 65535)
        self.assertEncode(b'\x80', 0, 64 * 1024)

    def test_encode(self):
        self.assertEncode(b'\x91\x01\x01', 1, 1)
        self.assertEncode(b'\x91\t\n', 9, 10)
        self.assertEncode(b'\x91\xfe\xff', 254, 255)
        self.assertEncode(b'\xa2\x02\x01', 512, 256)
        self.assertEncode(b'\xb3\x02\x01\x01\x01', 258, 257)
        self.assertEncode(b'\xb0\x01\x01', 0, 257)
        self.assertEncode(b'\x81\n', 10, 64 * 1024)

    def test_decode_no_length(self):
        self.assertDecode(0, 65536, 1, b'\x80', 0)
        self.assertDecode(1, 65536, 2, b'\x81\x01', 0)
        self.assertDecode(10, 65536, 2, b'\x81\n', 0)
        self.assertDecode(255, 65536, 2, b'\x81\xff', 0)
        self.assertDecode(256, 65536, 2, b'\x82\x01', 0)
        self.assertDecode(257, 65536, 3, b'\x83\x01\x01', 0)
        self.assertDecode(4294967295, 65536, 5, b'\x8f\xff\xff\xff\xff', 0)
        self.assertDecode(4294967040, 65536, 4, b'\x8e\xff\xff\xff', 0)
        self.assertDecode(4294902015, 65536, 4, b'\x8d\xff\xff\xff', 0)
        self.assertDecode(4278255615, 65536, 4, b'\x8b\xff\xff\xff', 0)
        self.assertDecode(16777215, 65536, 4, b'\x87\xff\xff\xff', 0)
        self.assertDecode(16909060, 65536, 5, b'\x8f\x04\x03\x02\x01', 0)

    def test_decode_no_offset(self):
        self.assertDecode(0, 1, 2, b'\x90\x01', 0)
        self.assertDecode(0, 10, 2, b'\x90\n', 0)
        self.assertDecode(0, 255, 2, b'\x90\xff', 0)
        self.assertDecode(0, 256, 2, b'\xa0\x01', 0)
        self.assertDecode(0, 257, 3, b'\xb0\x01\x01', 0)
        self.assertDecode(0, 65535, 3, b'\xb0\xff\xff', 0)
        self.assertDecode(0, 65536, 1, b'\x80', 0)

    def test_decode(self):
        self.assertDecode(1, 1, 3, b'\x91\x01\x01', 0)
        self.assertDecode(9, 10, 3, b'\x91\t\n', 0)
        self.assertDecode(254, 255, 3, b'\x91\xfe\xff', 0)
        self.assertDecode(512, 256, 3, b'\xa2\x02\x01', 0)
        self.assertDecode(258, 257, 5, b'\xb3\x02\x01\x01\x01', 0)
        self.assertDecode(0, 257, 3, b'\xb0\x01\x01', 0)

    def test_decode_not_start(self):
        self.assertDecode(1, 1, 6, b'abc\x91\x01\x01def', 3)
        self.assertDecode(9, 10, 5, b'ab\x91\t\nde', 2)
        self.assertDecode(254, 255, 6, b'not\x91\xfe\xffcopy', 3)