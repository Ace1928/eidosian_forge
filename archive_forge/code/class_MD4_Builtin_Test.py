from __future__ import with_statement, division
from binascii import hexlify
import hashlib
from passlib.utils.compat import bascii_to_str, PY3, u
from passlib.crypto.digest import lookup_hash
from passlib.tests.utils import TestCase, skipUnless
class MD4_Builtin_Test(_Common_MD4_Test):
    descriptionPrefix = 'passlib.crypto._md4.md4()'

    def setUp(self):
        super(MD4_Builtin_Test, self).setUp()
        if has_native_md4():
            orig = hashlib.new

            def wrapper(name, *args):
                if name == 'md4':
                    raise ValueError('md4 disabled for testing')
                return orig(name, *args)
            self.patchAttr(hashlib, 'new', wrapper)
            lookup_hash.clear_cache()
            self.addCleanup(lookup_hash.clear_cache)
        self.assertEqual(self.get_md4_const().__module__, 'passlib.crypto._md4')