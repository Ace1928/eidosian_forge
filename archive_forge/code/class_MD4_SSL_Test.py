from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
@skipUnless(has_native_md4(), 'hashlib lacks ssl/md4 support')
class MD4_SSL_Test(_Common_MD4_Test):
    descriptionPrefix = "hashlib.new('md4')"

    def setUp(self):
        super(MD4_SSL_Test, self).setUp()
        self.assertEqual(self.get_md4_const().__module__, 'hashlib')