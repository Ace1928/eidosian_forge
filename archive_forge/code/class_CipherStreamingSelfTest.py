import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
class CipherStreamingSelfTest(CipherSelfTest):

    def shortDescription(self):
        desc = self.module_name
        if self.mode is not None:
            desc += ' in %s mode' % (self.mode_name,)
        return '%s should behave like a stream cipher' % (desc,)

    def runTest(self):
        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)
        ct3 = []
        cipher = self._new()
        for i in range(0, len(plaintext), 3):
            ct3.append(cipher.encrypt(plaintext[i:i + 3]))
        ct3 = b2a_hex(b('').join(ct3))
        self.assertEqual(self.ciphertext, ct3)
        pt3 = []
        cipher = self._new()
        for i in range(0, len(ciphertext), 3):
            pt3.append(cipher.encrypt(ciphertext[i:i + 3]))
        pt3 = b2a_hex(b('').join(pt3))
        self.assertEqual(self.plaintext, pt3)