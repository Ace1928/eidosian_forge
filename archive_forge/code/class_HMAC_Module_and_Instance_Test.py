import unittest
from binascii import hexlify
from Cryptodome.Util.py3compat import tostr, tobytes
from Cryptodome.Hash import (HMAC, MD5, SHA1, SHA256,
class HMAC_Module_and_Instance_Test(unittest.TestCase):
    """Test the HMAC construction and verify that it does not
    matter if you initialize it with a hash module or
    with an hash instance.

    See https://bugs.launchpad.net/pycrypto/+bug/1209399
    """

    def __init__(self, hashmods):
        """Initialize the test with a dictionary of hash modules
        indexed by their names"""
        unittest.TestCase.__init__(self)
        self.hashmods = hashmods
        self.description = ''

    def shortDescription(self):
        return self.description

    def runTest(self):
        key = b'\x90\x91\x92\x93' * 4
        payload = b'\x00' * 100
        for hashname, hashmod in self.hashmods.items():
            if hashmod is None:
                continue
            self.description = 'Test HMAC in combination with ' + hashname
            one = HMAC.new(key, payload, hashmod).digest()
            two = HMAC.new(key, payload, hashmod.new()).digest()
            self.assertEqual(one, two)