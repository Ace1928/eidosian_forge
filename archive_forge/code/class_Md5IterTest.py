from Cryptodome.Util.py3compat import *
from Cryptodome.Hash import MD5
from binascii import unhexlify
import unittest
from Cryptodome.SelfTest.st_common import list_test_cases
class Md5IterTest(unittest.TestCase):

    def runTest(self):
        message = b('\x00') * 16
        result1 = '4AE71336E44BF9BF79D2752E234818A5'.lower()
        result2 = '1A83F51285E4D89403D00C46EF8508FE'.lower()
        h = MD5.new(message)
        message = h.digest()
        self.assertEqual(h.hexdigest(), result1)
        for _ in range(99999):
            h = MD5.new(message)
            message = h.digest()
        self.assertEqual(h.hexdigest(), result2)