import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import *
from Cryptodome.IO import PKCS8
from Cryptodome.Util.asn1 import DerNull
class Rng:

    def __init__(self, output):
        self.output = output
        self.idx = 0

    def __call__(self, n):
        output = self.output[self.idx:self.idx + n]
        self.idx += n
        return output