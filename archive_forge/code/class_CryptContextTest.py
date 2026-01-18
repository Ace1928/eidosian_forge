from __future__ import with_statement
from logging import getLogger
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, CryptPolicy, LazyCryptContext
from passlib.utils import to_bytes, to_unicode
import passlib.utils.handlers as uh
from passlib.tests.utils import TestCase, set_file
from passlib.registry import (register_crypt_handler_path,
class CryptContextTest(TestCase):
    """test CryptContext class"""
    descriptionPrefix = 'CryptContext'

    def setUp(self):
        TestCase.setUp(self)
        warnings.filterwarnings('ignore', 'CryptContext\\(\\)\\.replace\\(\\) has been deprecated.*')
        warnings.filterwarnings('ignore', 'The CryptContext ``policy`` keyword has been deprecated.*')
        warnings.filterwarnings('ignore', '.*(CryptPolicy|context\\.policy).*(has|have) been deprecated.*')
        warnings.filterwarnings('ignore', 'the method.*hash_needs_update.*is deprecated')

    def test_00_constructor(self):
        """test constructor"""
        cc = CryptContext([hash.md5_crypt, hash.bsdi_crypt, hash.des_crypt])
        c, b, a = cc.policy.iter_handlers()
        self.assertIs(a, hash.des_crypt)
        self.assertIs(b, hash.bsdi_crypt)
        self.assertIs(c, hash.md5_crypt)
        cc = CryptContext(['md5_crypt', 'bsdi_crypt', 'des_crypt'])
        c, b, a = cc.policy.iter_handlers()
        self.assertIs(a, hash.des_crypt)
        self.assertIs(b, hash.bsdi_crypt)
        self.assertIs(c, hash.md5_crypt)
        policy = cc.policy
        cc = CryptContext(policy=policy)
        self.assertEqual(cc.to_dict(), policy.to_dict())
        cc = CryptContext(policy=policy, default='bsdi_crypt')
        self.assertNotEqual(cc.to_dict(), policy.to_dict())
        self.assertEqual(cc.to_dict(), dict(schemes=['md5_crypt', 'bsdi_crypt', 'des_crypt'], default='bsdi_crypt'))
        self.assertRaises(TypeError, setattr, cc, 'policy', None)
        self.assertRaises(TypeError, CryptContext, policy='x')

    def test_01_replace(self):
        """test replace()"""
        cc = CryptContext(['md5_crypt', 'bsdi_crypt', 'des_crypt'])
        self.assertIs(cc.policy.get_handler(), hash.md5_crypt)
        cc2 = cc.replace()
        self.assertIsNot(cc2, cc)
        cc3 = cc.replace(default='bsdi_crypt')
        self.assertIsNot(cc3, cc)
        self.assertIs(cc3.policy.get_handler(), hash.bsdi_crypt)

    def test_02_no_handlers(self):
        """test no handlers"""
        cc = CryptContext()
        self.assertRaises(KeyError, cc.identify, 'hash', required=True)
        self.assertRaises(KeyError, cc.hash, 'secret')
        self.assertRaises(KeyError, cc.verify, 'secret', 'hash')
        cc = CryptContext(['md5_crypt'])
        p = CryptPolicy(schemes=[])
        cc.policy = p
        self.assertRaises(KeyError, cc.identify, 'hash', required=True)
        self.assertRaises(KeyError, cc.hash, 'secret')
        self.assertRaises(KeyError, cc.verify, 'secret', 'hash')
    sample_policy_1 = dict(schemes=['des_crypt', 'md5_crypt', 'phpass', 'bsdi_crypt', 'sha256_crypt'], deprecated=['des_crypt'], default='sha256_crypt', bsdi_crypt__max_rounds=30, bsdi_crypt__default_rounds=25, bsdi_crypt__vary_rounds=0, sha256_crypt__max_rounds=3000, sha256_crypt__min_rounds=2000, sha256_crypt__default_rounds=3000, phpass__ident='H', phpass__default_rounds=7)

    def test_12_hash_needs_update(self):
        """test hash_needs_update() method"""
        cc = CryptContext(**self.sample_policy_1)
        self.assertTrue(cc.hash_needs_update('9XXD4trGYeGJA'))
        self.assertFalse(cc.hash_needs_update('$1$J8HC2RCr$HcmM.7NxB2weSvlw2FgzU0'))
        self.assertTrue(cc.hash_needs_update('$5$rounds=1999$jD81UCoo.zI.UETs$Y7qSTQ6mTiU9qZB4fRr43wRgQq4V.5AAf7F97Pzxey/'))
        self.assertFalse(cc.hash_needs_update('$5$rounds=2000$228SSRje04cnNCaQ$YGV4RYu.5sNiBvorQDlO0WWQjyJVGKBcJXz3OtyQ2u8'))
        self.assertFalse(cc.hash_needs_update('$5$rounds=3000$fS9iazEwTKi7QPW4$VasgBC8FqlOvD7x2HhABaMXCTh9jwHclPA9j5YQdns.'))
        self.assertTrue(cc.hash_needs_update('$5$rounds=3001$QlFHHifXvpFX4PLs$/0ekt7lSs/lOikSerQ0M/1porEHxYq7W/2hdFpxA3fA'))

    def test_30_nonstring_hash(self):
        """test non-string hash values cause error"""
        warnings.filterwarnings('ignore', ".*needs_update.*'scheme' keyword is deprecated.*")
        cc = CryptContext(['des_crypt'])
        for hash, kwds in [(None, {}), (None, {'scheme': 'des_crypt'}), (1, {}), ((), {})]:
            self.assertRaises(TypeError, cc.hash_needs_update, hash, **kwds)
        cc2 = CryptContext(['mysql323'])
        self.assertRaises(TypeError, cc2.hash_needs_update, None)