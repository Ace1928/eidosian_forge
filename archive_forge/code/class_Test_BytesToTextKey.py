from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
class Test_BytesToTextKey(tests.TestCase):

    def assertBytesToTextKey(self, key, bytes):
        self.assertEqual(key, self.module._bytes_to_text_key(bytes))

    def assertBytesToTextKeyRaises(self, bytes):
        self.assertRaises(Exception, self.module._bytes_to_text_key, bytes)

    def test_file(self):
        self.assertBytesToTextKey((b'file-id', b'revision-id'), b'file: file-id\nparent-id\nname\nrevision-id\nda39a3ee5e6b4b0d3255bfef95601890afd80709\n100\nN')

    def test_invalid_no_kind(self):
        self.assertBytesToTextKeyRaises(b'file  file-id\nparent-id\nname\nrevision-id\nda39a3ee5e6b4b0d3255bfef95601890afd80709\n100\nN')

    def test_invalid_no_space(self):
        self.assertBytesToTextKeyRaises(b'file:file-id\nparent-id\nname\nrevision-id\nda39a3ee5e6b4b0d3255bfef95601890afd80709\n100\nN')

    def test_invalid_too_short_file_id(self):
        self.assertBytesToTextKeyRaises(b'file:file-id')

    def test_invalid_too_short_parent_id(self):
        self.assertBytesToTextKeyRaises(b'file:file-id\nparent-id')

    def test_invalid_too_short_name(self):
        self.assertBytesToTextKeyRaises(b'file:file-id\nparent-id\nname')

    def test_dir(self):
        self.assertBytesToTextKey((b'dir-id', b'revision-id'), b'dir: dir-id\nparent-id\nname\nrevision-id')