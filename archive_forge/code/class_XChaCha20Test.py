import os
import re
import unittest
from binascii import hexlify, unhexlify
from Cryptodome.Util.py3compat import b, tobytes, bchr
from Cryptodome.Util.strxor import strxor_c
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Cipher import ChaCha20
class XChaCha20Test(unittest.TestCase):

    def test_hchacha20(self):
        from Cryptodome.Cipher.ChaCha20 import _HChaCha20
        key = b'00:01:02:03:04:05:06:07:08:09:0a:0b:0c:0d:0e:0f:10:11:12:13:14:15:16:17:18:19:1a:1b:1c:1d:1e:1f'
        key = unhexlify(key.replace(b':', b''))
        nonce = b'00:00:00:09:00:00:00:4a:00:00:00:00:31:41:59:27'
        nonce = unhexlify(nonce.replace(b':', b''))
        subkey = _HChaCha20(key, nonce)
        expected = b'82413b42 27b27bfe d30e4250 8a877d73 a0f9e4d5 8a74a853 c12ec413 26d3ecdc'
        expected = unhexlify(expected.replace(b' ', b''))
        self.assertEqual(subkey, expected)

    def test_nonce(self):
        key = b'A' * 32
        nonce = b'P' * 24
        cipher = ChaCha20.new(key=key, nonce=nonce)
        self.assertEqual(nonce, cipher.nonce)

    def test_encrypt(self):
        pt = b'\n                5468652064686f6c65202870726f6e6f756e6365642022646f6c652229206973\n                20616c736f206b6e6f776e2061732074686520417369617469632077696c6420\n                646f672c2072656420646f672c20616e642077686973746c696e6720646f672e\n                2049742069732061626f7574207468652073697a65206f662061204765726d61\n                6e20736865706865726420627574206c6f6f6b73206d6f7265206c696b652061\n                206c6f6e672d6c656767656420666f782e205468697320686967686c7920656c\n                757369766520616e6420736b696c6c6564206a756d70657220697320636c6173\n                736966696564207769746820776f6c7665732c20636f796f7465732c206a6163\n                6b616c732c20616e6420666f78657320696e20746865207461786f6e6f6d6963\n                2066616d696c792043616e696461652e'
        pt = unhexlify(pt.replace(b'\n', b'').replace(b' ', b''))
        key = unhexlify(b'808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f')
        iv = unhexlify(b'404142434445464748494a4b4c4d4e4f5051525354555658')
        ct = b'\n                7d0a2e6b7f7c65a236542630294e063b7ab9b555a5d5149aa21e4ae1e4fbce87\n                ecc8e08a8b5e350abe622b2ffa617b202cfad72032a3037e76ffdcdc4376ee05\n                3a190d7e46ca1de04144850381b9cb29f051915386b8a710b8ac4d027b8b050f\n                7cba5854e028d564e453b8a968824173fc16488b8970cac828f11ae53cabd201\n                12f87107df24ee6183d2274fe4c8b1485534ef2c5fbc1ec24bfc3663efaa08bc\n                047d29d25043532db8391a8a3d776bf4372a6955827ccb0cdd4af403a7ce4c63\n                d595c75a43e045f0cce1f29c8b93bd65afc5974922f214a40b7c402cdb91ae73\n                c0b63615cdad0480680f16515a7ace9d39236464328a37743ffc28f4ddb324f4\n                d0f5bbdc270c65b1749a6efff1fbaa09536175ccd29fb9e6057b307320d31683\n                8a9c71f70b5b5907a66f7ea49aadc409'
        ct = unhexlify(ct.replace(b'\n', b'').replace(b' ', b''))
        cipher = ChaCha20.new(key=key, nonce=iv)
        cipher.seek(64)
        ct_test = cipher.encrypt(pt)
        self.assertEqual(ct, ct_test)