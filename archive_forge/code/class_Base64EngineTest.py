from __future__ import with_statement
from functools import partial
import warnings
from passlib.utils import is_ascii_safe, to_bytes
from passlib.utils.compat import irange, PY2, PY3, u, unicode, join_bytes, PYPY
from passlib.tests.utils import TestCase, hb, run_with_fixed_seeds
from passlib.utils.binary import h64, h64big
class Base64EngineTest(TestCase):
    """test standalone parts of Base64Engine"""

    def test_constructor(self):
        from passlib.utils.binary import Base64Engine, AB64_CHARS
        self.assertRaises(TypeError, Base64Engine, 1)
        self.assertRaises(ValueError, Base64Engine, AB64_CHARS[:-1])
        self.assertRaises(ValueError, Base64Engine, AB64_CHARS[:-1] + 'A')

    def test_ab64_decode(self):
        """ab64_decode()"""
        from passlib.utils.binary import ab64_decode
        self.assertEqual(ab64_decode(b'abc'), hb('69b7'))
        self.assertEqual(ab64_decode(u('abc')), hb('69b7'))
        self.assertRaises(ValueError, ab64_decode, u('abÿ'))
        self.assertRaises(TypeError, ab64_decode, b'ab\xff')
        self.assertRaises(TypeError, ab64_decode, b'ab!')
        self.assertRaises(TypeError, ab64_decode, u('ab!'))
        self.assertEqual(ab64_decode(b'abcd'), hb('69b71d'))
        self.assertRaises(ValueError, ab64_decode, b'abcde')
        self.assertEqual(ab64_decode(b'abcdef'), hb('69b71d79'))
        self.assertEqual(ab64_decode(b'abcdeQ'), hb('69b71d79'))
        self.assertEqual(ab64_decode(b'abcdefg'), hb('69b71d79f8'))
        self.assertEqual(ab64_decode(b'ab+/'), hb('69bfbf'))
        self.assertEqual(ab64_decode(b'ab./'), hb('69bfbf'))

    def test_ab64_encode(self):
        """ab64_encode()"""
        from passlib.utils.binary import ab64_encode
        self.assertEqual(ab64_encode(hb('69b7')), b'abc')
        self.assertRaises(TypeError if PY3 else UnicodeEncodeError, ab64_encode, hb('69b7').decode('latin-1'))
        self.assertEqual(ab64_encode(hb('69b71d')), b'abcd')
        self.assertEqual(ab64_encode(hb('69b71d79')), b'abcdeQ')
        self.assertEqual(ab64_encode(hb('69b71d79f8')), b'abcdefg')
        self.assertEqual(ab64_encode(hb('69bfbf')), b'ab./')

    def test_b64s_decode(self):
        """b64s_decode()"""
        from passlib.utils.binary import b64s_decode
        self.assertEqual(b64s_decode(b'abc'), hb('69b7'))
        self.assertEqual(b64s_decode(u('abc')), hb('69b7'))
        self.assertRaises(ValueError, b64s_decode, u('abÿ'))
        self.assertRaises(TypeError, b64s_decode, b'ab\xff')
        self.assertRaises(TypeError, b64s_decode, b'ab!')
        self.assertRaises(TypeError, b64s_decode, u('ab!'))
        self.assertEqual(b64s_decode(b'abcd'), hb('69b71d'))
        self.assertRaises(ValueError, b64s_decode, b'abcde')
        self.assertEqual(b64s_decode(b'abcdef'), hb('69b71d79'))
        self.assertEqual(b64s_decode(b'abcdeQ'), hb('69b71d79'))
        self.assertEqual(b64s_decode(b'abcdefg'), hb('69b71d79f8'))

    def test_b64s_encode(self):
        """b64s_encode()"""
        from passlib.utils.binary import b64s_encode
        self.assertEqual(b64s_encode(hb('69b7')), b'abc')
        self.assertRaises(TypeError if PY3 else UnicodeEncodeError, b64s_encode, hb('69b7').decode('latin-1'))
        self.assertEqual(b64s_encode(hb('69b71d')), b'abcd')
        self.assertEqual(b64s_encode(hb('69b71d79')), b'abcdeQ')
        self.assertEqual(b64s_encode(hb('69b71d79f8')), b'abcdefg')
        self.assertEqual(b64s_encode(hb('69bfbf')), b'ab+/')