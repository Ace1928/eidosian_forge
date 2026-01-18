import unittest
from binascii import a2b_hex, b2a_hex, hexlify
from Cryptodome.Util.py3compat import b
from Cryptodome.Util.strxor import strxor_c
class ByteArrayTest(unittest.TestCase):
    """Verify we can use bytearray's for encrypting and decrypting"""

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        params = params.copy()
        self.description = _extract(params, 'description')
        self.key = b(_extract(params, 'key'))
        self.plaintext = b(_extract(params, 'plaintext'))
        self.ciphertext = b(_extract(params, 'ciphertext'))
        self.module_name = _extract(params, 'module_name', None)
        self.assoc_data = _extract(params, 'assoc_data', None)
        self.mac = _extract(params, 'mac', None)
        if self.assoc_data:
            self.mac = b(self.mac)
        mode = _extract(params, 'mode', None)
        self.mode_name = str(mode)
        if mode is not None:
            self.mode = getattr(self.module, 'MODE_' + mode)
            self.iv = _extract(params, 'iv', None)
            if self.iv is None:
                self.iv = _extract(params, 'nonce', None)
            if self.iv is not None:
                self.iv = b(self.iv)
        else:
            self.mode = None
            self.iv = _extract(params, 'iv', None)
            if self.iv is not None:
                self.iv = b(self.iv)
        self.extra_params = params

    def _new(self):
        params = self.extra_params.copy()
        key = a2b_hex(self.key)
        old_style = []
        if self.mode is not None:
            old_style = [self.mode]
        if self.iv is not None:
            old_style += [a2b_hex(self.iv)]
        return self.module.new(key, *old_style, **params)

    def runTest(self):
        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)
        assoc_data = []
        if self.assoc_data:
            assoc_data = [bytearray(a2b_hex(b(x))) for x in self.assoc_data]
        cipher = self._new()
        decipher = self._new()
        for comp in assoc_data:
            cipher.update(comp)
            decipher.update(comp)
        ct = b2a_hex(cipher.encrypt(bytearray(plaintext)))
        pt = b2a_hex(decipher.decrypt(bytearray(ciphertext)))
        self.assertEqual(self.ciphertext, ct)
        self.assertEqual(self.plaintext, pt)
        if self.mac:
            mac = b2a_hex(cipher.digest())
            self.assertEqual(self.mac, mac)
            decipher.verify(bytearray(a2b_hex(self.mac)))