from __future__ import print_function
import unittest
from Cryptodome.PublicKey import RSA
from Cryptodome.SelfTest.st_common import list_test_cases, a2b_hex
from Cryptodome import Random
from Cryptodome.Cipher import PKCS1_v1_5 as PKCS
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
class PKCS1_15_Tests(unittest.TestCase):

    def setUp(self):
        self.rng = Random.new().read
        self.key1024 = RSA.generate(1024, self.rng)
    _testData = (('-----BEGIN RSA PRIVATE KEY-----\nMIICXAIBAAKBgQDAiAnvIAOvqVwJTaYzsKnefZftgtXGE2hPJppGsWl78yz9jeXY\nW/FxX/gTPURArNhdnhP6n3p2ZaDIBrO2zizbgIXs0IsljTTcr4vnI8fMXzyNUOjA\nzP3nzMqZDZK6757XQAobOssMkBFqRWwilT/3DsBhRpl3iMUhF+wvpTSHewIDAQAB\nAoGAC4HV/inOrpgTvSab8Wj0riyZgQOZ3U3ZpSlsfR8ra9Ib9Uee3jCYnKscu6Gk\ny6zI/cdt8EPJ4PuwAWSNJzbpbVaDvUq25OD+CX8/uRT08yBS4J8TzBitZJTD4lS7\natdTnKT0Wmwk+u8tDbhvMKwnUHdJLcuIsycts9rwJVapUtkCQQDvDpx2JMun0YKG\nuUttjmL8oJ3U0m3ZvMdVwBecA0eebZb1l2J5PvI3EJD97eKe91Nsw8T3lwpoN40k\nIocSVDklAkEAzi1HLHE6EzVPOe5+Y0kGvrIYRRhncOb72vCvBZvD6wLZpQgqo6c4\nd3XHFBBQWA6xcvQb5w+VVEJZzw64y25sHwJBAMYReRl6SzL0qA0wIYrYWrOt8JeQ\n8mthulcWHXmqTgC6FEXP9Es5GD7/fuKl4wqLKZgIbH4nqvvGay7xXLCXD/ECQH9a\n1JYNMtRen5unSAbIOxRcKkWz92F0LKpm9ZW/S9vFHO+mBcClMGoKJHiuQxLBsLbT\nNtEZfSJZAeS2sUtn3/0CQDb2M2zNBTF8LlM0nxmh0k9VGm5TVIyBEMcipmvOgqIs\nHKukWBcq9f/UOmS0oEhai/6g+Uf7VHJdWaeO5LzuvwU=\n-----END RSA PRIVATE KEY-----', 'THIS IS PLAINTEXT\n', '3f dc fd 3c cd 5c 9b 12  af 65 32 e3 f7 d0 da 36\n                8f 8f d9 e3 13 1c 7f c8  b3 f9 c1 08 e4 eb 79 9c\n                91 89 1f 96 3b 94 77 61  99 a4 b1 ee 5d e6 17 c9\n                5d 0a b5 63 52 0a eb 00  45 38 2a fb b0 71 3d 11\n                f7 a1 9e a7 69 b3 af 61  c0 bb 04 5b 5d 4b 27 44\n                1f 5b 97 89 ba 6a 08 95  ee 4f a2 eb 56 64 e5 0f\n                da 7c f9 9a 61 61 06 62  ed a0 bc 5f aa 6c 31 78\n                70 28 1a bb 98 3c e3 6a  60 3c d1 0b 0f 5a f4 75', 'eb d7 7d 86 a4 35 23 a3 54 7e 02 0b 42 1d\n                61 6c af 67 b8 4e 17 56 80 66 36 04 64 34 26 8a\n                47 dd 44 b3 1a b2 17 60 f4 91 2e e2 b5 95 64 cc\n                f9 da c8 70 94 54 86 4c ef 5b 08 7d 18 c4 ab 8d\n                04 06 33 8f ca 15 5f 52 60 8a a1 0c f5 08 b5 4c\n                bb 99 b8 94 25 04 9c e6 01 75 e6 f9 63 7a 65 61\n                13 8a a7 47 77 81 ae 0d b8 2c 4d 50 a5'),)

    def testEncrypt1(self):
        for test in self._testData:
            key = RSA.importKey(test[0])

            class randGen:

                def __init__(self, data):
                    self.data = data
                    self.idx = 0

                def __call__(self, N):
                    r = self.data[self.idx:self.idx + N]
                    self.idx += N
                    return r
            cipher = PKCS.new(key, randfunc=randGen(t2b(test[3])))
            ct = cipher.encrypt(b(test[1]))
            self.assertEqual(ct, t2b(test[2]))

    def testEncrypt2(self):
        pt = '\x00' * (128 - 11 + 1)
        cipher = PKCS.new(self.key1024)
        self.assertRaises(ValueError, cipher.encrypt, pt)

    def testVerify1(self):
        for test in self._testData:
            key = RSA.importKey(test[0])
            expected_pt = b(test[1])
            ct = t2b(test[2])
            cipher = PKCS.new(key)
            pt = cipher.decrypt(ct, None)
            self.assertEqual(pt, expected_pt)
            pt = cipher.decrypt(ct, b'\xff' * len(expected_pt))
            self.assertEqual(pt, expected_pt)

    def testVerify2(self):
        cipher = PKCS.new(self.key1024)
        self.assertRaises(ValueError, cipher.decrypt, '\x00' * 127, '---')
        self.assertRaises(ValueError, cipher.decrypt, '\x00' * 129, '---')
        pt = b('\x00\x02' + 'ÿ' * 7 + '\x00' + 'E' * 118)
        pt_int = bytes_to_long(pt)
        ct_int = self.key1024._encrypt(pt_int)
        ct = long_to_bytes(ct_int, 128)
        self.assertEqual(b'---', cipher.decrypt(ct, b'---'))

    def testEncryptVerify1(self):
        for pt_len in range(0, 128 - 11 + 1):
            pt = self.rng(pt_len)
            cipher = PKCS.new(self.key1024)
            ct = cipher.encrypt(pt)
            pt2 = cipher.decrypt(ct, b'\xaa' * pt_len)
            self.assertEqual(pt, pt2)

    def test_encrypt_verify_exp_pt_len(self):
        cipher = PKCS.new(self.key1024)
        pt = b'5' * 16
        ct = cipher.encrypt(pt)
        sentinel = b'\xaa' * 16
        pt_A = cipher.decrypt(ct, sentinel, 16)
        self.assertEqual(pt, pt_A)
        pt_B = cipher.decrypt(ct, sentinel, 15)
        self.assertEqual(sentinel, pt_B)
        pt_C = cipher.decrypt(ct, sentinel, 17)
        self.assertEqual(sentinel, pt_C)

    def testByteArray(self):
        pt = b'XER'
        cipher = PKCS.new(self.key1024)
        ct = cipher.encrypt(bytearray(pt))
        pt2 = cipher.decrypt(bytearray(ct), 'ÿ' * len(pt))
        self.assertEqual(pt, pt2)

    def testMemoryview(self):
        pt = b'XER'
        cipher = PKCS.new(self.key1024)
        ct = cipher.encrypt(memoryview(bytearray(pt)))
        pt2 = cipher.decrypt(memoryview(bytearray(ct)), b'\xff' * len(pt))
        self.assertEqual(pt, pt2)

    def test_return_type(self):
        pt = b'XYZ'
        cipher = PKCS.new(self.key1024)
        ct = cipher.encrypt(pt)
        self.assertTrue(isinstance(ct, bytes))
        pt2 = cipher.decrypt(ct, b'\xaa' * 3)
        self.assertTrue(isinstance(pt2, bytes))