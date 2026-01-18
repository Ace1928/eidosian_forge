import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
class TestFloresta(unittest.TestCase):

    def test_words(self):
        words = floresta.words()[:10]
        txt = 'Um revivalismo refrescante O 7_e_Meio é um ex-libris de a'
        self.assertEqual(words, txt.split())