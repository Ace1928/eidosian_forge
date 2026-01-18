import pathlib
import unittest
from pygame import encode_string, encode_file_path
class RWopsEncodeStringTest(unittest.TestCase):
    global getrefcount

    def test_obj_None(self):
        encoded_string = encode_string(None)
        self.assertIsNone(encoded_string)

    def test_returns_bytes(self):
        u = 'Hello'
        encoded_string = encode_string(u)
        self.assertIsInstance(encoded_string, bytes)

    def test_obj_bytes(self):
        b = b'encyclop\xe6dia'
        encoded_string = encode_string(b, 'ascii', 'strict')
        self.assertIs(encoded_string, b)

    def test_encode_unicode(self):
        u = 'Þe Olde Kompüter Shoppe'
        b = u.encode('utf-8')
        self.assertEqual(encode_string(u, 'utf-8'), b)

    def test_error_fowarding(self):
        self.assertRaises(SyntaxError, encode_string)

    def test_errors(self):
        u = 'abcĉdefgĝhĥijĵklmnoprsŝtuŭvz'
        b = u.encode('ascii', 'ignore')
        self.assertEqual(encode_string(u, 'ascii', 'ignore'), b)

    def test_encoding_error(self):
        u = 'a\x80b'
        encoded_string = encode_string(u, 'ascii', 'strict')
        self.assertIsNone(encoded_string)

    def test_check_defaults(self):
        u = 'aǷb'
        b = u.encode('unicode_escape', 'backslashreplace')
        encoded_string = encode_string(u)
        self.assertEqual(encoded_string, b)

    def test_etype(self):
        u = 'a\x80b'
        self.assertRaises(SyntaxError, encode_string, u, 'ascii', 'strict', SyntaxError)

    def test_etype__invalid(self):
        """Ensures invalid etypes are properly handled."""
        for etype in ('SyntaxError', self):
            self.assertRaises(TypeError, encode_string, 'test', etype=etype)

    def test_string_with_null_bytes(self):
        b = b'a\x00b\x00c'
        encoded_string = encode_string(b, etype=SyntaxError)
        encoded_decode_string = encode_string(b.decode(), 'ascii', 'strict')
        self.assertIs(encoded_string, b)
        self.assertEqual(encoded_decode_string, b)
    try:
        from sys import getrefcount as _g
        getrefcount = _g
    except ImportError:
        pass
    else:

        def test_refcount(self):
            bpath = b' This is a string that is not cached.'[1:]
            upath = bpath.decode('ascii')
            before = getrefcount(bpath)
            bpath = encode_string(bpath)
            self.assertEqual(getrefcount(bpath), before)
            bpath = encode_string(upath)
            self.assertEqual(getrefcount(bpath), before)

    def test_smp(self):
        utf_8 = b'a\xf0\x93\x82\xa7b'
        u = 'a𓂧b'
        b = encode_string(u, 'utf-8', 'strict', AssertionError)
        self.assertEqual(b, utf_8)

    def test_pathlib_obj(self):
        """Test loading string representation of pathlib object"""
        '\n        We do this because pygame functions internally use pg_EncodeString\n        to decode the filenames passed to them. So if we test that here, we\n        can safely assume that all those functions do not have any issues\n        with pathlib objects\n        '
        encoded = encode_string(pathlib.PurePath('foo'), 'utf-8')
        self.assertEqual(encoded, b'foo')
        encoded = encode_string(pathlib.Path('baz'))
        self.assertEqual(encoded, b'baz')