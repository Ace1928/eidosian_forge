import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import sys
import time as _time
from passlib import exc
from passlib.utils.compat import unicode, u
from passlib.tests.utils import TestCase, time_call
from passlib import totp as totp_module
from passlib.totp import TOTP, AppWallet, AES_SUPPORT
class AppWalletTest(TestCase):
    descriptionPrefix = 'passlib.totp.AppWallet'

    def test_secrets_types(self):
        """constructor -- 'secrets' param -- input types"""
        wallet = AppWallet()
        self.assertEqual(wallet._secrets, {})
        self.assertFalse(wallet.has_secrets)
        ref = {'1': b'aaa', '2': b'bbb'}
        wallet = AppWallet(ref)
        self.assertEqual(wallet._secrets, ref)
        self.assertTrue(wallet.has_secrets)
        wallet = AppWallet('\n 1: aaa\n# comment\n \n2: bbb   ')
        self.assertEqual(wallet._secrets, ref)
        wallet = AppWallet('1: aaa: bbb \n# comment\n \n2: bbb   ')
        self.assertEqual(wallet._secrets, {'1': b'aaa: bbb', '2': b'bbb'})
        wallet = AppWallet('{"1":"aaa","2":"bbb"}')
        self.assertEqual(wallet._secrets, ref)
        self.assertRaises(TypeError, AppWallet, 123)
        self.assertRaises(TypeError, AppWallet, '[123]')
        self.assertRaises(ValueError, AppWallet, {'1': 'aaa', '2': ''})

    def test_secrets_tags(self):
        """constructor -- 'secrets' param -- tag/value normalization"""
        ref = {'1': b'aaa', '02': b'bbb', 'C': b'ccc'}
        wallet = AppWallet(ref)
        self.assertEqual(wallet._secrets, ref)
        wallet = AppWallet({u('1'): b'aaa', u('02'): b'bbb', u('C'): b'ccc'})
        self.assertEqual(wallet._secrets, ref)
        wallet = AppWallet({1: b'aaa', '02': b'bbb', 'C': b'ccc'})
        self.assertEqual(wallet._secrets, ref)
        self.assertRaises(TypeError, AppWallet, {(1,): 'aaa'})
        wallet = AppWallet({'1-2_3.4': b'aaa'})
        self.assertRaises(ValueError, AppWallet, {'-abc': 'aaa'})
        self.assertRaises(ValueError, AppWallet, {'ab*$': 'aaa'})
        wallet = AppWallet({'1': u('aaa'), '02': 'bbb', 'C': b'ccc'})
        self.assertEqual(wallet._secrets, ref)
        self.assertRaises(TypeError, AppWallet, {'1': 123})
        self.assertRaises(TypeError, AppWallet, {'1': None})
        self.assertRaises(TypeError, AppWallet, {'1': []})

    def test_default_tag(self):
        """constructor -- 'default_tag' param"""
        wallet = AppWallet({'1': 'one', '02': 'two'})
        self.assertEqual(wallet.default_tag, '02')
        self.assertEqual(wallet.get_secret(wallet.default_tag), b'two')
        wallet = AppWallet({'1': 'one', '02': 'two', 'A': 'aaa'})
        self.assertEqual(wallet.default_tag, 'A')
        self.assertEqual(wallet.get_secret(wallet.default_tag), b'aaa')
        wallet = AppWallet({'1': 'one', '02': 'two', 'A': 'aaa'}, default_tag='1')
        self.assertEqual(wallet.default_tag, '1')
        self.assertEqual(wallet.get_secret(wallet.default_tag), b'one')
        self.assertRaises(KeyError, AppWallet, {'1': 'one', '02': 'two', 'A': 'aaa'}, default_tag='B')
        wallet = AppWallet()
        self.assertEqual(wallet.default_tag, None)
        self.assertRaises(KeyError, wallet.get_secret, None)

    def require_aes_support(self, canary=None):
        if AES_SUPPORT:
            canary and canary()
        else:
            canary and self.assertRaises(RuntimeError, canary)
            raise self.skipTest("'cryptography' package not installed")

    def test_decrypt_key(self):
        """.decrypt_key()"""
        wallet = AppWallet({'1': PASS1, '2': PASS2})
        CIPHER1 = dict(v=1, c=13, s='6D7N7W53O7HHS37NLUFQ', k='MHCTEGSNPFN5CGBJ', t='1')
        self.require_aes_support(canary=partial(wallet.decrypt_key, CIPHER1))
        self.assertEqual(wallet.decrypt_key(CIPHER1)[0], KEY1_RAW)
        CIPHER2 = dict(v=1, c=13, s='SPZJ54Y6IPUD2BYA4C6A', k='ZGDXXTVQOWYLC2AU', t='1')
        self.assertEqual(wallet.decrypt_key(CIPHER2)[0], KEY1_RAW)
        CIPHER3 = dict(v=1, c=8, s='FCCTARTIJWE7CPQHUDKA', k='D2DRS32YESGHHINWFFCELKN7Z6NAHM4M', t='2')
        self.assertEqual(wallet.decrypt_key(CIPHER3)[0], KEY2_RAW)
        temp = CIPHER1.copy()
        temp.update(t='2')
        self.assertEqual(wallet.decrypt_key(temp)[0], b'\xafD6.F7\xeb\x19\x05Q')
        temp = CIPHER1.copy()
        temp.update(t='3')
        self.assertRaises(KeyError, wallet.decrypt_key, temp)
        temp = CIPHER1.copy()
        temp.update(v=999)
        self.assertRaises(ValueError, wallet.decrypt_key, temp)

    def test_decrypt_key_needs_recrypt(self):
        """.decrypt_key() -- needs_recrypt flag"""
        self.require_aes_support()
        wallet = AppWallet({'1': PASS1, '2': PASS2}, encrypt_cost=13)
        ref = dict(v=1, c=13, s='AAAA', k='AAAA', t='2')
        self.assertFalse(wallet.decrypt_key(ref)[1])
        temp = ref.copy()
        temp.update(c=8)
        self.assertTrue(wallet.decrypt_key(temp)[1])
        temp = ref.copy()
        temp.update(t='1')
        self.assertTrue(wallet.decrypt_key(temp)[1])

    def assertSaneResult(self, result, wallet, key, tag='1', needs_recrypt=False):
        """check encrypt_key() result has expected format"""
        self.assertEqual(set(result), set(['v', 't', 'c', 's', 'k']))
        self.assertEqual(result['v'], 1)
        self.assertEqual(result['t'], tag)
        self.assertEqual(result['c'], wallet.encrypt_cost)
        self.assertEqual(len(result['s']), to_b32_size(wallet.salt_size))
        self.assertEqual(len(result['k']), to_b32_size(len(key)))
        result_key, result_needs_recrypt = wallet.decrypt_key(result)
        self.assertEqual(result_key, key)
        self.assertEqual(result_needs_recrypt, needs_recrypt)

    def test_encrypt_key(self):
        """.encrypt_key()"""
        wallet = AppWallet({'1': PASS1}, encrypt_cost=5)
        self.require_aes_support(canary=partial(wallet.encrypt_key, KEY1_RAW))
        result = wallet.encrypt_key(KEY1_RAW)
        self.assertSaneResult(result, wallet, KEY1_RAW)
        other = wallet.encrypt_key(KEY1_RAW)
        self.assertSaneResult(result, wallet, KEY1_RAW)
        self.assertNotEqual(other['s'], result['s'])
        self.assertNotEqual(other['k'], result['k'])
        wallet2 = AppWallet({'1': PASS1}, encrypt_cost=6)
        result = wallet2.encrypt_key(KEY1_RAW)
        self.assertSaneResult(result, wallet2, KEY1_RAW)
        wallet2 = AppWallet({'1': PASS1, '2': PASS2})
        result = wallet2.encrypt_key(KEY1_RAW)
        self.assertSaneResult(result, wallet2, KEY1_RAW, tag='2')
        wallet2 = AppWallet({'1': PASS1})
        wallet2.salt_size = 64
        result = wallet2.encrypt_key(KEY1_RAW)
        self.assertSaneResult(result, wallet2, KEY1_RAW)
        result = wallet.encrypt_key(KEY2_RAW)
        self.assertSaneResult(result, wallet, KEY2_RAW)
        self.assertRaises(ValueError, wallet.encrypt_key, b'')

    def test_encrypt_cost_timing(self):
        """verify cost parameter via timing"""
        self.require_aes_support()
        wallet = AppWallet({'1': 'aaa'})
        wallet.encrypt_cost -= 2
        delta, _ = time_call(partial(wallet.encrypt_key, KEY1_RAW), maxtime=0)
        wallet.encrypt_cost += 3
        delta2, _ = time_call(partial(wallet.encrypt_key, KEY1_RAW), maxtime=0)
        self.assertAlmostEqual(delta2, delta * 8, delta=delta * 8 * 0.5)