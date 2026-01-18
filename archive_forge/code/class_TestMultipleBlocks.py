from __future__ import print_function
import unittest
from Cryptodome.Hash import SHA256
from Cryptodome.Cipher import AES
from Cryptodome.Util.py3compat import *
from binascii import hexlify
class TestMultipleBlocks(unittest.TestCase):

    def __init__(self, use_aesni):
        unittest.TestCase.__init__(self)
        self.use_aesni = use_aesni

    def runTest(self):
        tvs = [(b'a' * 16, 'c0b27011eb15bf144d2fc9fae80ea16d4c231cb230416c5fac02e6835ad9d7d0'), (b'a' * 24, 'df8435ce361a78c535b41dcb57da952abbf9ee5954dc6fbcd75fd00fa626915d'), (b'a' * 32, '211402de6c80db1f92ba255881178e1f70783b8cfd3b37808205e48b80486cd8')]
        for key, expected in tvs:
            cipher = AES.new(key, AES.MODE_ECB, use_aesni=self.use_aesni)
            h = SHA256.new()
            pt = b''.join([tobytes('{0:016x}'.format(x)) for x in range(20)])
            ct = cipher.encrypt(pt)
            self.assertEqual(SHA256.new(ct).hexdigest(), expected)