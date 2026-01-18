import logging
import os
import tempfile
import time
from hashlib import sha256
from tests.unit import unittest
from boto.compat import BytesIO, six, StringIO
from boto.glacier.utils import minimum_part_size, chunk_hashes, tree_hash, \
class TestChunking(unittest.TestCase):

    def test_chunk_hashes_exact(self):
        chunks = chunk_hashes(b'a' * (2 * 1024 * 1024))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], sha256(b'a' * 1024 * 1024).digest())

    def test_chunks_with_leftovers(self):
        bytestring = b'a' * (2 * 1024 * 1024 + 20)
        chunks = chunk_hashes(bytestring)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], sha256(b'a' * 1024 * 1024).digest())
        self.assertEqual(chunks[1], sha256(b'a' * 1024 * 1024).digest())
        self.assertEqual(chunks[2], sha256(b'a' * 20).digest())

    def test_less_than_one_chunk(self):
        chunks = chunk_hashes(b'aaaa')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], sha256(b'aaaa').digest())