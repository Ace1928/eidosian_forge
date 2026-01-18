from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary
class BM25ABCTest(unittest.TestCase):

    def setUp(self):
        self.documents = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        self.dictionary = Dictionary(self.documents)
        self.expected_avgdl = sum(map(len, self.documents)) / len(self.documents)

    def test_avgdl_from_corpus(self):
        corpus = list(map(self.dictionary.doc2bow, self.documents))
        model = BM25Stub(corpus=corpus)
        actual_avgdl = model.avgdl
        self.assertAlmostEqual(self.expected_avgdl, actual_avgdl)

    def test_avgdl_from_dictionary(self):
        model = BM25Stub(dictionary=self.dictionary)
        actual_avgdl = model.avgdl
        self.assertAlmostEqual(self.expected_avgdl, actual_avgdl)