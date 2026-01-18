import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
class TestPartSizeCalculations(unittest.TestCase):

    def test_small_values_still_use_default_part_size(self):
        self.assertEqual(minimum_part_size(1), 4 * 1024 * 1024)

    def test_under_the_maximum_value(self):
        self.assertEqual(minimum_part_size(8 * 1024 * 1024), 4 * 1024 * 1024)

    def test_gigabyte_size(self):
        self.assertEqual(minimum_part_size(8 * 1024 * 1024 * 10000), 8 * 1024 * 1024)

    def test_terabyte_size(self):
        self.assertEqual(minimum_part_size(4 * 1024 * 1024 * 1024 * 1024), 512 * 1024 * 1024)

    def test_file_size_too_large(self):
        with self.assertRaises(ValueError):
            minimum_part_size(40000 * 1024 * 1024 * 1024 + 1)

    def test_default_part_size_can_be_specified(self):
        default_part_size = 2 * 1024 * 1024
        self.assertEqual(minimum_part_size(8 * 1024 * 1024, default_part_size), default_part_size)