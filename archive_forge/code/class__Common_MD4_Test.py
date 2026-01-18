from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
class _Common_MD4_Test(TestCase):
    """common code for testing md4 backends"""
    vectors = [(b'', '31d6cfe0d16ae931b73c59d7e0c089c0'), (b'a', 'bde52cb31de33e46245e05fbdbd6fb24'), (b'abc', 'a448017aaf21d8525fc10ae87aa6729d'), (b'message digest', 'd9130a8164549fe818874806e1c7014b'), (b'abcdefghijklmnopqrstuvwxyz', 'd79e1c308aa5bbcdeea8ed63df412da9'), (b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789', '043f8582f241db351ce627e153e7f0e4'), (b'12345678901234567890123456789012345678901234567890123456789012345678901234567890', 'e33b4ddc9c38f2199c3e7b164fcc0536')]

    def get_md4_const(self):
        """
        get md4 constructor --
        overridden by subclasses to use alternate backends.
        """
        return lookup_hash('md4').const

    def test_attrs(self):
        """informational attributes"""
        h = self.get_md4_const()()
        self.assertEqual(h.name, 'md4')
        self.assertEqual(h.digest_size, 16)
        self.assertEqual(h.block_size, 64)

    def test_md4_update(self):
        """update() method"""
        md4 = self.get_md4_const()
        h = md4(b'')
        self.assertEqual(h.hexdigest(), '31d6cfe0d16ae931b73c59d7e0c089c0')
        h.update(b'a')
        self.assertEqual(h.hexdigest(), 'bde52cb31de33e46245e05fbdbd6fb24')
        h.update(b'bcdefghijklmnopqrstuvwxyz')
        self.assertEqual(h.hexdigest(), 'd79e1c308aa5bbcdeea8ed63df412da9')
        if PY3:
            h = md4()
            self.assertRaises(TypeError, h.update, u('a'))
            self.assertEqual(h.hexdigest(), '31d6cfe0d16ae931b73c59d7e0c089c0')
        else:
            h = md4()
            h.update(u('a'))
            self.assertEqual(h.hexdigest(), 'bde52cb31de33e46245e05fbdbd6fb24')

    def test_md4_hexdigest(self):
        """hexdigest() method"""
        md4 = self.get_md4_const()
        for input, hex in self.vectors:
            out = md4(input).hexdigest()
            self.assertEqual(out, hex)

    def test_md4_digest(self):
        """digest() method"""
        md4 = self.get_md4_const()
        for input, hex in self.vectors:
            out = bascii_to_str(hexlify(md4(input).digest()))
            self.assertEqual(out, hex)

    def test_md4_copy(self):
        """copy() method"""
        md4 = self.get_md4_const()
        h = md4(b'abc')
        h2 = h.copy()
        h2.update(b'def')
        self.assertEqual(h2.hexdigest(), '804e7f1c2586e50b49ac65db5b645131')
        h.update(b'ghi')
        self.assertEqual(h.hexdigest(), 'c5225580bfe176f6deeee33dee98732c')