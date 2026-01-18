from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
class NgramsTest(unittest.TestCase):

    def setUp(self):
        self.expected_text = {'test': ['<te', 'tes', 'est', 'st>', '<tes', 'test', 'est>', '<test', 'test>'], 'at the': ['<at', 'at ', 't t', ' th', 'the', 'he>', '<at ', 'at t', 't th', ' the', 'the>', '<at t', 'at th', 't the', ' the>'], 'at\nthe': ['<at', 'at\n', 't\nt', '\nth', 'the', 'he>', '<at\n', 'at\nt', 't\nth', '\nthe', 'the>', '<at\nt', 'at\nth', 't\nthe', '\nthe>'], 'тест': ['<те', 'тес', 'ест', 'ст>', '<тес', 'тест', 'ест>', '<тест', 'тест>'], 'テスト': ['<テス', 'テスト', 'スト>', '<テスト', 'テスト>', '<テスト>'], '試し': ['<試し', '試し>', '<試し>']}
        self.expected_bytes = {'test': [b'<te', b'<tes', b'<test', b'tes', b'test', b'test>', b'est', b'est>', b'st>'], 'at the': [b'<at', b'<at ', b'<at t', b'at ', b'at t', b'at th', b't t', b't th', b't the', b' th', b' the', b' the>', b'the', b'the>', b'he>'], 'тест': [b'<\xd1\x82\xd0\xb5', b'<\xd1\x82\xd0\xb5\xd1\x81', b'<\xd1\x82\xd0\xb5\xd1\x81\xd1\x82', b'\xd1\x82\xd0\xb5\xd1\x81', b'\xd1\x82\xd0\xb5\xd1\x81\xd1\x82', b'\xd1\x82\xd0\xb5\xd1\x81\xd1\x82>', b'\xd0\xb5\xd1\x81\xd1\x82', b'\xd0\xb5\xd1\x81\xd1\x82>', b'\xd1\x81\xd1\x82>'], 'テスト': [b'<\xe3\x83\x86\xe3\x82\xb9', b'<\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88', b'<\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88>', b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88', b'\xe3\x83\x86\xe3\x82\xb9\xe3\x83\x88>', b'\xe3\x82\xb9\xe3\x83\x88>'], '試し': [b'<\xe8\xa9\xa6\xe3\x81\x97', b'<\xe8\xa9\xa6\xe3\x81\x97>', b'\xe8\xa9\xa6\xe3\x81\x97>']}
        self.expected_text_wide_unicode = {'🚑🚒🚓🚕': ['<🚑🚒', '🚑🚒🚓', '🚒🚓🚕', '🚓🚕>', '<🚑🚒🚓', '🚑🚒🚓🚕', '🚒🚓🚕>', '<🚑🚒🚓🚕', '🚑🚒🚓🚕>']}
        self.expected_bytes_wide_unicode = {'🚑🚒🚓🚕': [b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92', b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93', b'<\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95', b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93', b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95', b'\xf0\x9f\x9a\x91\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>', b'\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95', b'\xf0\x9f\x9a\x92\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>', b'\xf0\x9f\x9a\x93\xf0\x9f\x9a\x95>']}

    def test_text_cy(self):
        for word in self.expected_text:
            expected = self.expected_text[word]
            actual = compute_ngrams(word, 3, 5)
            self.assertEqual(expected, actual)

    @unittest.skipIf(sys.maxunicode == 65535, "Python interpreter doesn't support UCS-4 (wide unicode)")
    def test_text_cy_wide_unicode(self):
        for word in self.expected_text_wide_unicode:
            expected = self.expected_text_wide_unicode[word]
            actual = compute_ngrams(word, 3, 5)
            self.assertEqual(expected, actual)

    def test_bytes_cy(self):
        for word in self.expected_bytes:
            expected = self.expected_bytes[word]
            actual = compute_ngrams_bytes(word, 3, 5)
            self.assertEqual(expected, actual)
            expected_text = self.expected_text[word]
            actual_text = [n.decode('utf-8') for n in actual]
            self.assertEqual(sorted(expected_text), sorted(actual_text))
        for word in self.expected_bytes_wide_unicode:
            expected = self.expected_bytes_wide_unicode[word]
            actual = compute_ngrams_bytes(word, 3, 5)
            self.assertEqual(expected, actual)
            expected_text = self.expected_text_wide_unicode[word]
            actual_text = [n.decode('utf-8') for n in actual]
            self.assertEqual(sorted(expected_text), sorted(actual_text))

    def test_fb(self):
        """Test against results from Facebook's implementation."""
        with utils.open(datapath('fb-ngrams.txt'), 'r', encoding='utf-8') as fin:
            fb = dict(_read_fb(fin))
        for word, expected in fb.items():
            actual = compute_ngrams(word, 3, 6)
            self.assertEqual(sorted(expected), sorted(actual))