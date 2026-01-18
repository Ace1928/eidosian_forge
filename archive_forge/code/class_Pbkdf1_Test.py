from __future__ import with_statement, division
from binascii import hexlify
import hashlib
import warnings
from passlib.exc import UnknownHashError
from passlib.utils.compat import PY3, u, JYTHON
from passlib.tests.utils import TestCase, TEST_MODE, skipUnless, hb
from passlib.crypto.digest import pbkdf2_hmac, PBKDF2_BACKENDS
class Pbkdf1_Test(TestCase):
    """test kdf helpers"""
    descriptionPrefix = 'passlib.crypto.digest.pbkdf1'
    pbkdf1_tests = [(b'password', hb('78578E5A5D63CB06'), 1000, 16, 'sha1', hb('dc19847e05c64d2faf10ebfb4a3d2a20')), (b'password', b'salt', 1000, 0, 'md5', b''), (b'password', b'salt', 1000, 1, 'md5', hb('84')), (b'password', b'salt', 1000, 8, 'md5', hb('8475c6a8531a5d27')), (b'password', b'salt', 1000, 16, 'md5', hb('8475c6a8531a5d27e386cd496457812c')), (b'password', b'salt', 1000, None, 'md5', hb('8475c6a8531a5d27e386cd496457812c')), (b'password', b'salt', 1000, None, 'sha1', hb('4a8fd48e426ed081b535be5769892fa396293efb'))]
    if not JYTHON:
        pbkdf1_tests.append((b'password', b'salt', 1000, None, 'md4', hb('f7f2e91100a8f96190f2dd177cb26453')))

    def test_known(self):
        """test reference vectors"""
        from passlib.crypto.digest import pbkdf1
        for secret, salt, rounds, keylen, digest, correct in self.pbkdf1_tests:
            result = pbkdf1(digest, secret, salt, rounds, keylen)
            self.assertEqual(result, correct)

    def test_border(self):
        """test border cases"""
        from passlib.crypto.digest import pbkdf1

        def helper(secret=b'secret', salt=b'salt', rounds=1, keylen=1, hash='md5'):
            return pbkdf1(hash, secret, salt, rounds, keylen)
        helper()
        self.assertRaises(TypeError, helper, secret=1)
        self.assertRaises(TypeError, helper, salt=1)
        self.assertRaises(ValueError, helper, hash='missing')
        self.assertRaises(ValueError, helper, rounds=0)
        self.assertRaises(TypeError, helper, rounds='1')
        self.assertRaises(ValueError, helper, keylen=-1)
        self.assertRaises(ValueError, helper, keylen=17, hash='md5')
        self.assertRaises(TypeError, helper, keylen='1')