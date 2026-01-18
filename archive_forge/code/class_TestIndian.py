import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
class TestIndian(unittest.TestCase):

    def test_words(self):
        words = indian.words()[:3]
        self.assertEqual(words, ['মহিষের', 'সন্তান', ':'])

    def test_tagged_words(self):
        tagged_words = indian.tagged_words()[:3]
        self.assertEqual(tagged_words, [('মহিষের', 'NN'), ('সন্তান', 'NN'), (':', 'SYM')])