import unittest
from Cryptodome.IO._PBES import PBES2
class TestPBES2(unittest.TestCase):

    def setUp(self):
        self.ref = b'Test data'
        self.passphrase = b'Passphrase'

    def test1(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test2(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA224AndAES128-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test3(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA256AndAES192-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test4(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA384AndAES256-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test5(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA512AndAES128-GCM')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test6(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA512-224AndAES192-GCM')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test7(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'PBKDF2WithHMAC-SHA3-256AndAES256-GCM')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test8(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'scryptAndAES128-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test9(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'scryptAndAES192-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test10(self):
        ct = PBES2.encrypt(self.ref, self.passphrase, 'scryptAndAES256-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)