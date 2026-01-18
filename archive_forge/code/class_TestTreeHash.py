import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
class TestTreeHash(unittest.TestCase):

    def calculate_tree_hash(self, bytestring):
        start = time.time()
        calculated = bytes_to_hex(tree_hash(chunk_hashes(bytestring)))
        end = time.time()
        logging.debug('Tree hash calc time for length %s: %s', len(bytestring), end - start)
        return calculated

    def test_tree_hash_calculations(self):
        one_meg_bytestring = b'a' * (1 * 1024 * 1024)
        two_meg_bytestring = b'a' * (2 * 1024 * 1024)
        four_meg_bytestring = b'a' * (4 * 1024 * 1024)
        bigger_bytestring = four_meg_bytestring + b'a' * 20
        self.assertEqual(self.calculate_tree_hash(one_meg_bytestring), b'9bc1b2a288b26af7257a36277ae3816a7d4f16e89c1e7e77d0a5c48bad62b360')
        self.assertEqual(self.calculate_tree_hash(two_meg_bytestring), b'560c2c9333c719cb00cfdffee3ba293db17f58743cdd1f7e4055373ae6300afa')
        self.assertEqual(self.calculate_tree_hash(four_meg_bytestring), b'9491cb2ed1d4e7cd53215f4017c23ec4ad21d7050a1e6bb636c4f67e8cddb844')
        self.assertEqual(self.calculate_tree_hash(bigger_bytestring), b'12f3cbd6101b981cde074039f6f728071da8879d6f632de8afc7cdf00661b08f')

    def test_empty_tree_hash(self):
        self.assertEqual(self.calculate_tree_hash(''), b'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')