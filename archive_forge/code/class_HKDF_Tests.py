import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
class HKDF_Tests(unittest.TestCase):
    _test_vector = ((SHA256, '0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b', '000102030405060708090a0b0c', 'f0f1f2f3f4f5f6f7f8f9', 42, '3cb25f25faacd57a90434f64d0362f2a' + '2d2d0a90cf1a5a4c5db02d56ecc4c5bf' + '34007208d5b887185865'), (SHA256, '000102030405060708090a0b0c0d0e0f' + '101112131415161718191a1b1c1d1e1f' + '202122232425262728292a2b2c2d2e2f' + '303132333435363738393a3b3c3d3e3f' + '404142434445464748494a4b4c4d4e4f', '606162636465666768696a6b6c6d6e6f' + '707172737475767778797a7b7c7d7e7f' + '808182838485868788898a8b8c8d8e8f' + '909192939495969798999a9b9c9d9e9f' + 'a0a1a2a3a4a5a6a7a8a9aaabacadaeaf', 'b0b1b2b3b4b5b6b7b8b9babbbcbdbebf' + 'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf' + 'd0d1d2d3d4d5d6d7d8d9dadbdcdddedf' + 'e0e1e2e3e4e5e6e7e8e9eaebecedeeef' + 'f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff', 82, 'b11e398dc80327a1c8e7f78c596a4934' + '4f012eda2d4efad8a050cc4c19afa97c' + '59045a99cac7827271cb41c65e590e09' + 'da3275600c2f09b8367793a9aca3db71' + 'cc30c58179ec3e87c14c01d5c1f3434f' + '1d87'), (SHA256, '0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b', None, None, 42, '8da4e775a563c18f715f802a063c5a31' + 'b8a11f5c5ee1879ec3454e5f3c738d2d' + '9d201395faa4b61a96c8'), (SHA1, '0b0b0b0b0b0b0b0b0b0b0b', '000102030405060708090a0b0c', 'f0f1f2f3f4f5f6f7f8f9', 42, '085a01ea1b10f36933068b56efa5ad81' + 'a4f14b822f5b091568a9cdd4f155fda2' + 'c22e422478d305f3f896'), (SHA1, '000102030405060708090a0b0c0d0e0f' + '101112131415161718191a1b1c1d1e1f' + '202122232425262728292a2b2c2d2e2f' + '303132333435363738393a3b3c3d3e3f' + '404142434445464748494a4b4c4d4e4f', '606162636465666768696a6b6c6d6e6f' + '707172737475767778797a7b7c7d7e7f' + '808182838485868788898a8b8c8d8e8f' + '909192939495969798999a9b9c9d9e9f' + 'a0a1a2a3a4a5a6a7a8a9aaabacadaeaf', 'b0b1b2b3b4b5b6b7b8b9babbbcbdbebf' + 'c0c1c2c3c4c5c6c7c8c9cacbcccdcecf' + 'd0d1d2d3d4d5d6d7d8d9dadbdcdddedf' + 'e0e1e2e3e4e5e6e7e8e9eaebecedeeef' + 'f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff', 82, '0bd770a74d1160f7c9f12cd5912a06eb' + 'ff6adcae899d92191fe4305673ba2ffe' + '8fa3f1a4e5ad79f3f334b3b202b2173c' + '486ea37ce3d397ed034c7f9dfeb15c5e' + '927336d0441f4c4300e2cff0d0900b52' + 'd3b4'), (SHA1, '0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b', '', '', 42, '0ac1af7002b3d761d1e55298da9d0506' + 'b9ae52057220a306e07b6b87e8df21d0' + 'ea00033de03984d34918'), (SHA1, '0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c', None, '', 42, '2c91117204d745f3500d636a62f64f0a' + 'b3bae548aa53d423b0d1f27ebba6f5e5' + '673a081d70cce7acfc48'))

    def test1(self):
        for tv in self._test_vector:
            secret, salt, info, exp = [t2b(tv[x]) for x in (1, 2, 3, 5)]
            key_len, hashmod = [tv[x] for x in (4, 0)]
            output = HKDF(secret, key_len, salt, hashmod, 1, info)
            self.assertEqual(output, exp)

    def test2(self):
        ref = HKDF(b('XXXXXX'), 12, b('YYYY'), SHA1)
        key1, key2 = HKDF(b('XXXXXX'), 6, b('YYYY'), SHA1, 2)
        self.assertEqual((ref[:6], ref[6:]), (key1, key2))
        key1, key2, key3 = HKDF(b('XXXXXX'), 4, b('YYYY'), SHA1, 3)
        self.assertEqual((ref[:4], ref[4:8], ref[8:]), (key1, key2, key3))