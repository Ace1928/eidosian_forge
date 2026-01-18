import json
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512, SHA3_384,
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pkcs1_15
from Cryptodome.Signature import PKCS1_v1_5
from Cryptodome.Util._file_system import pycryptodome_filename
from Cryptodome.Util.strxor import strxor
class PKCS1_15_NoParams(unittest.TestCase):
    """Verify that PKCS#1 v1.5 signatures pass even without NULL parameters in
    the algorithm identifier (PyCrypto/LP bug #1119552)."""
    rsakey = '-----BEGIN RSA PRIVATE KEY-----\n            MIIBOwIBAAJBAL8eJ5AKoIsjURpcEoGubZMxLD7+kT+TLr7UkvEtFrRhDDKMtuII\n            q19FrL4pUIMymPMSLBn3hJLe30Dw48GQM4UCAwEAAQJACUSDEp8RTe32ftq8IwG8\n            Wojl5mAd1wFiIOrZ/Uv8b963WJOJiuQcVN29vxU5+My9GPZ7RA3hrDBEAoHUDPrI\n            OQIhAPIPLz4dphiD9imAkivY31Rc5AfHJiQRA7XixTcjEkojAiEAyh/pJHks/Mlr\n            +rdPNEpotBjfV4M4BkgGAA/ipcmaAjcCIQCHvhwwKVBLzzTscT2HeUdEeBMoiXXK\n            JACAr3sJQJGxIQIgarRp+m1WSKV1MciwMaTOnbU7wxFs9DP1pva76lYBzgUCIQC9\n            n0CnZCJ6IZYqSt0H5N7+Q+2Ro64nuwV/OSQfM6sBwQ==\n            -----END RSA PRIVATE KEY-----'
    msg = b'This is a test\n'
    sig_str = 'a287a13517f716e72fb14eea8e33a8db4a4643314607e7ca3e3e281893db74013dda8b855fd99f6fecedcb25fcb7a434f35cd0a101f8b19348e0bd7b6f152dfc'
    signature = unhexlify(sig_str)

    def runTest(self):
        verifier = pkcs1_15.new(RSA.importKey(self.rsakey))
        hashed = SHA1.new(self.msg)
        verifier.verify(hashed, self.signature)