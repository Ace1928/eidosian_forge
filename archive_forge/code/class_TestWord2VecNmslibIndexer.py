import logging
import unittest
import math
import os
import numpy
import scipy
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import word2vec
from gensim.models import doc2vec
from gensim.models import KeyedVectors
from gensim.models import TfidfModel
from gensim import matutils, similarities
from gensim.models import Word2Vec, FastText
from gensim.test.utils import (
from gensim.similarities import UniformTermSimilarityIndex
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import LevenshteinSimilarityIndex
from gensim.similarities.docsim import _nlargest
from gensim.similarities.fastss import editdist
class TestWord2VecNmslibIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import nmslib
        except ImportError as e:
            raise unittest.SkipTest('NMSLIB library is not available: %s' % e)
        from gensim.similarities.nmslib import NmslibIndexer
        self.indexer = NmslibIndexer

    def test_word2vec(self):
        model = word2vec.Word2Vec(TEXTS, min_count=1)
        index = self.indexer(model)
        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_fasttext(self):

        class LeeReader:

            def __init__(self, fn):
                self.fn = fn

            def __iter__(self):
                with utils.open(self.fn, 'r', encoding='latin_1') as infile:
                    for line in infile:
                        yield line.lower().strip().split()
        model = FastText(LeeReader(datapath('lee.cor')), bucket=5000)
        index = self.indexer(model)
        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_indexing_keyedvectors(self):
        from gensim.similarities.nmslib import NmslibIndexer
        keyVectors_file = datapath('lee_fasttext.vec')
        model = KeyedVectors.load_word2vec_format(keyVectors_file)
        index = NmslibIndexer(model)
        self.assertVectorIsSimilarToItself(model, index)
        self.assertApproxNeighborsMatchExact(model, model, index)

    def test_load_missing_raises_error(self):
        from gensim.similarities.nmslib import NmslibIndexer
        self.assertRaises(IOError, NmslibIndexer.load, fname='test-index')

    def assertVectorIsSimilarToItself(self, wv, index):
        vector = wv.get_normed_vectors()[0]
        label = wv.index_to_key[0]
        approx_neighbors = index.most_similar(vector, 1)
        word, similarity = approx_neighbors[0]
        self.assertEqual(word, label)
        self.assertAlmostEqual(similarity, 1.0, places=2)

    def assertApproxNeighborsMatchExact(self, model, wv, index):
        vector = wv.get_normed_vectors()[0]
        approx_neighbors = model.most_similar([vector], topn=5, indexer=index)
        exact_neighbors = model.most_similar([vector], topn=5)
        approx_words = [word_id for word_id, similarity in approx_neighbors]
        exact_words = [word_id for word_id, similarity in exact_neighbors]
        self.assertEqual(approx_words, exact_words)

    def assertIndexSaved(self, index):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def assertLoadedIndexEqual(self, index, model):
        from gensim.similarities.nmslib import NmslibIndexer
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        index2 = NmslibIndexer.load(fname)
        index2.model = model
        self.assertEqual(index.labels, index2.labels)
        self.assertEqual(index.index_params, index2.index_params)
        self.assertEqual(index.query_time_params, index2.query_time_params)