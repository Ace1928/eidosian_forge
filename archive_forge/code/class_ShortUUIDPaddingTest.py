import os
import string
import sys
import unittest
from collections import defaultdict
from unittest.mock import patch
from uuid import UUID
from uuid import uuid4
from shortuuid.cli import cli
from shortuuid.main import decode
from shortuuid.main import encode
from shortuuid.main import get_alphabet
from shortuuid.main import random
from shortuuid.main import set_alphabet
from shortuuid.main import ShortUUID
from shortuuid.main import uuid
class ShortUUIDPaddingTest(unittest.TestCase):

    def test_padding(self):
        su = ShortUUID()
        random_uid = uuid4()
        smallest_uid = UUID(int=0)
        encoded_random = su.encode(random_uid)
        encoded_small = su.encode(smallest_uid)
        self.assertEqual(len(encoded_random), len(encoded_small))

    def test_decoding(self):
        su = ShortUUID()
        random_uid = uuid4()
        smallest_uid = UUID(int=0)
        encoded_random = su.encode(random_uid)
        encoded_small = su.encode(smallest_uid)
        self.assertEqual(su.decode(encoded_small), smallest_uid)
        self.assertEqual(su.decode(encoded_random), random_uid)

    def test_consistency(self):
        su = ShortUUID()
        num_iterations = 1000
        uid_lengths = defaultdict(int)
        for count in range(num_iterations):
            random_uid = uuid4()
            encoded_random = su.encode(random_uid)
            uid_lengths[len(encoded_random)] += 1
            decoded_random = su.decode(encoded_random)
            self.assertEqual(random_uid, decoded_random)
        self.assertEqual(len(uid_lengths), 1)
        uid_length = next(iter(uid_lengths.keys()))
        self.assertEqual(uid_lengths[uid_length], num_iterations)