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
class TestWord2VecAnnoyIndexer(unittest.TestCase):

    def setUp(self):
        try:
            import annoy
        except ImportError as e:
            raise unittest.SkipTest('Annoy library is not available: %s' % e)
        from gensim.similarities.annoy import AnnoyIndexer
        self.indexer = AnnoyIndexer

    def test_word2vec(self):
        model = word2vec.Word2Vec(TEXTS, min_count=1)
        index = self.indexer(model, 10)
        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_fast_text(self):

        class LeeReader:

            def __init__(self, fn):
                self.fn = fn

            def __iter__(self):
                with utils.open(self.fn, 'r', encoding='latin_1') as infile:
                    for line in infile:
                        yield line.lower().strip().split()
        model = FastText(LeeReader(datapath('lee.cor')), bucket=5000)
        index = self.indexer(model, 10)
        self.assertVectorIsSimilarToItself(model.wv, index)
        self.assertApproxNeighborsMatchExact(model.wv, model.wv, index)
        self.assertIndexSaved(index)
        self.assertLoadedIndexEqual(index, model)

    def test_annoy_indexing_of_keyed_vectors(self):
        from gensim.similarities.annoy import AnnoyIndexer
        keyVectors_file = datapath('lee_fasttext.vec')
        model = KeyedVectors.load_word2vec_format(keyVectors_file)
        index = AnnoyIndexer(model, 10)
        self.assertEqual(index.num_trees, 10)
        self.assertVectorIsSimilarToItself(model, index)
        self.assertApproxNeighborsMatchExact(model, model, index)

    def test_load_missing_raises_error(self):
        from gensim.similarities.annoy import AnnoyIndexer
        test_index = AnnoyIndexer()
        self.assertRaises(IOError, test_index.load, fname='test-index')

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
        exact_neighbors = model.most_similar(positive=[vector], topn=5)
        approx_words = [neighbor[0] for neighbor in approx_neighbors]
        exact_words = [neighbor[0] for neighbor in exact_neighbors]
        self.assertEqual(approx_words, exact_words)

    def assertAllSimilaritiesDisableIndexer(self, model, wv, index):
        vector = wv.get_normed_vectors()[0]
        approx_similarities = model.most_similar([vector], topn=None, indexer=index)
        exact_similarities = model.most_similar(positive=[vector], topn=None)
        self.assertEqual(approx_similarities, exact_similarities)
        self.assertEqual(len(approx_similarities), len(wv.vectors))

    def assertIndexSaved(self, index):
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname + '.d'))

    def assertLoadedIndexEqual(self, index, model):
        from gensim.similarities.annoy import AnnoyIndexer
        fname = get_tmpfile('gensim_similarities.tst.pkl')
        index.save(fname)
        index2 = AnnoyIndexer()
        index2.load(fname)
        index2.model = model
        self.assertEqual(index.index.f, index2.index.f)
        self.assertEqual(index.labels, index2.labels)
        self.assertEqual(index.num_trees, index2.num_trees)