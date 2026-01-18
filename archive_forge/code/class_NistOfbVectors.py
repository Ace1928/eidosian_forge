import unittest
from binascii import unhexlify
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.Util.py3compat import tobytes
from Cryptodome.Cipher import AES, DES3, DES
from Cryptodome.Hash import SHAKE128
from Cryptodome.SelfTest.loader import load_test_vectors_wycheproof
from Cryptodome.SelfTest.Cipher.test_CBC import BlockChainingTests
from Cryptodome.SelfTest.Cipher.test_CBC import NistBlockChainingVectors
class NistOfbVectors(NistBlockChainingVectors):
    aes_mode = AES.MODE_OFB
    des_mode = DES.MODE_OFB
    des3_mode = DES3.MODE_OFB