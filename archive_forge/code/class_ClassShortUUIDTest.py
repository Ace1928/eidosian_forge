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
class ClassShortUUIDTest(unittest.TestCase):

    def test_generation(self):
        su = ShortUUID()
        self.assertTrue(20 < len(su.uuid()) < 24)
        self.assertTrue(20 < len(su.uuid('http://www.example.com/')) < 24)
        self.assertTrue(20 < len(su.uuid('HTTP://www.example.com/')) < 24)
        self.assertTrue(20 < len(su.uuid('example.com/')) < 24)

    def test_encoding(self):
        su = ShortUUID()
        u = UUID('{3b1f8b40-222c-4a6e-b77e-779d5a94e21c}')
        self.assertEqual(su.encode(u), 'CXc85b4rqinB7s5J52TRYb')

    def test_decoding(self):
        su = ShortUUID()
        u = UUID('{3b1f8b40-222c-4a6e-b77e-779d5a94e21c}')
        self.assertEqual(su.decode('CXc85b4rqinB7s5J52TRYb'), u)

    def test_random(self):
        su = ShortUUID()
        for i in range(1000):
            self.assertEqual(len(su.random()), 22)
        for i in range(1, 100):
            self.assertEqual(len(su.random(i)), i)

    def test_alphabet(self):
        alphabet = '01'
        su1 = ShortUUID(alphabet)
        su2 = ShortUUID()
        self.assertEqual(alphabet, su1.get_alphabet())
        su1.set_alphabet('01010101010101')
        self.assertEqual(alphabet, su1.get_alphabet())
        self.assertEqual(set(su1.uuid()), set('01'))
        self.assertTrue(116 < len(su1.uuid()) < 140)
        self.assertTrue(20 < len(su2.uuid()) < 24)
        u = uuid4()
        self.assertEqual(u, su1.decode(su1.encode(u)))
        u = su1.uuid()
        self.assertEqual(u, su1.encode(su1.decode(u)))
        self.assertRaises(ValueError, su1.set_alphabet, '1')
        self.assertRaises(ValueError, su1.set_alphabet, '1111111')

    def test_encoded_length(self):
        su1 = ShortUUID()
        self.assertEqual(su1.encoded_length(), 22)
        base64_alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + '+/'
        su2 = ShortUUID(base64_alphabet)
        self.assertEqual(su2.encoded_length(), 22)
        binary_alphabet = '01'
        su3 = ShortUUID(binary_alphabet)
        self.assertEqual(su3.encoded_length(), 128)
        su4 = ShortUUID()
        self.assertEqual(su4.encoded_length(num_bytes=8), 11)