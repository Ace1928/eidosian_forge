import unittest
from Cryptodome.Util.py3compat import *
class LargeSHA256Test(unittest.TestCase):

    def runTest(self):
        """SHA256: 512/520 MiB test"""
        from Cryptodome.Hash import SHA256
        zeros = bchr(0) * (1024 * 1024)
        h = SHA256.new(zeros)
        for i in range(511):
            h.update(zeros)
        self.assertEqual('9acca8e8c22201155389f65abbf6bc9723edc7384ead80503839f49dcc56d767', h.hexdigest())
        for i in range(8):
            h.update(zeros)
        self.assertEqual('abf51ad954b246009dfe5a50ecd582fd5b8f1b8b27f30393853c3ef721e7fa6e', h.hexdigest())