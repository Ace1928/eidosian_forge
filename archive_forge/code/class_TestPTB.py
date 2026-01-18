import unittest
import pytest
from nltk.corpus import (  # mwa_ppdb
from nltk.tree import Tree
@pytest.mark.skipif(not ptb.fileids(), reason='A full installation of the Penn Treebank is not available')
class TestPTB(unittest.TestCase):

    def test_fileids(self):
        self.assertEqual(ptb.fileids()[:4], ['BROWN/CF/CF01.MRG', 'BROWN/CF/CF02.MRG', 'BROWN/CF/CF03.MRG', 'BROWN/CF/CF04.MRG'])

    def test_words(self):
        self.assertEqual(ptb.words('WSJ/00/WSJ_0003.MRG')[:7], ['A', 'form', 'of', 'asbestos', 'once', 'used', '*'])

    def test_tagged_words(self):
        self.assertEqual(ptb.tagged_words('WSJ/00/WSJ_0003.MRG')[:3], [('A', 'DT'), ('form', 'NN'), ('of', 'IN')])

    def test_categories(self):
        self.assertEqual(ptb.categories(), ['adventure', 'belles_lettres', 'fiction', 'humor', 'lore', 'mystery', 'news', 'romance', 'science_fiction'])

    def test_news_fileids(self):
        self.assertEqual(ptb.fileids('news')[:3], ['WSJ/00/WSJ_0001.MRG', 'WSJ/00/WSJ_0002.MRG', 'WSJ/00/WSJ_0003.MRG'])

    def test_category_words(self):
        self.assertEqual(ptb.words(categories=['humor', 'fiction'])[:6], ['Thirty-three', 'Scotty', 'did', 'not', 'go', 'back'])