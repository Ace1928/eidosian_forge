import unittest
import copy
import logging
import numbers
import numpy as np
from gensim import matutils
from gensim.models import nmf
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
class TestNmf(unittest.TestCase, basetmtests.TestBaseTopicModel):

    def setUp(self):
        self.model = nmf.Nmf(common_corpus, id2word=common_dictionary, chunksize=1, num_topics=2, passes=100, random_state=42)

    def test_generator(self):
        model_1 = nmf.Nmf(iter(common_corpus * 100), id2word=common_dictionary, chunksize=1, num_topics=2, passes=1, random_state=42)
        model_2 = nmf.Nmf(common_corpus * 100, id2word=common_dictionary, chunksize=1, num_topics=2, passes=1, random_state=42)
        self.assertTrue(np.allclose(model_1.get_topics(), model_2.get_topics()))

    def test_update(self):
        model = copy.deepcopy(self.model)
        model.update(common_corpus)
        self.assertFalse(np.allclose(self.model.get_topics(), model.get_topics()))

    def test_random_state(self):
        model_1 = nmf.Nmf(common_corpus, id2word=common_dictionary, chunksize=1, num_topics=2, passes=100, random_state=42)
        model_2 = nmf.Nmf(common_corpus, id2word=common_dictionary, chunksize=1, num_topics=2, passes=100, random_state=0)
        self.assertTrue(np.allclose(self.model.get_topics(), model_1.get_topics()))
        self.assertFalse(np.allclose(self.model.get_topics(), model_2.get_topics()))

    def test_transform(self):
        doc = list(common_corpus)[0]
        transformed = self.model[doc]
        vec = matutils.sparse2full(transformed, 2)
        expected = [0.03028875, 0.96971124]
        self.assertTrue(np.allclose(sorted(vec), sorted(expected), atol=0.001))
        word = 5
        transformed = self.model.get_term_topics(word)
        vec = matutils.sparse2full(transformed, 2)
        expected = [[0.3076869, 0.69231313]]
        self.assertTrue(np.allclose(sorted(vec), sorted(expected), atol=0.001))

    def test_top_topics(self):
        top_topics = self.model.top_topics(common_corpus)
        for topic, score in top_topics:
            self.assertTrue(isinstance(topic, list))
            self.assertTrue(isinstance(score, float))
            for v, k in topic:
                self.assertTrue(isinstance(k, str))
                self.assertTrue(np.issubdtype(v, float))

    def test_get_topic_terms(self):
        topic_terms = self.model.get_topic_terms(1)
        for k, v in topic_terms:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, float))

    def test_get_document_topics(self):
        doc_topics = self.model.get_document_topics(common_corpus)
        for topic in doc_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, float))
        all_topics = self.model.get_document_topics(common_corpus)
        print(list(all_topics))
        for topic in all_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, float))

    def test_term_topics(self):
        result = self.model.get_term_topics(2)
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(np.issubdtype(probability, float))
        result = self.model.get_term_topics(str(self.model.id2word[2]))
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(np.issubdtype(probability, float))

    def test_persistence(self):
        fname = get_tmpfile('gensim_models_nmf.tst')
        self.model.save(fname)
        model2 = nmf.Nmf.load(fname)
        tstvec = []
        self.assertTrue(np.allclose(self.model[tstvec], model2[tstvec]))

    def test_large_mmap(self):
        fname = get_tmpfile('gensim_models_nmf.tst')
        self.model.save(fname, sep_limit=0)
        model2 = nmf.Nmf.load(fname, mmap='r')
        self.assertEqual(self.model.num_topics, model2.num_topics)
        tstvec = []
        self.assertTrue(np.allclose(self.model[tstvec], model2[tstvec]))

    def test_large_mmap_compressed(self):
        fname = get_tmpfile('gensim_models_nmf.tst.gz')
        self.model.save(fname, sep_limit=0)
        self.assertRaises(IOError, nmf.Nmf.load, fname, mmap='r')

    def test_dtype_backward_compatibility(self):
        nmf_fname = datapath('nmf_model')
        test_doc = [(0, 1), (1, 1), (2, 1)]
        expected_topics = [(1, 1.0)]
        model = nmf.Nmf.load(nmf_fname)
        topics = model[test_doc]
        self.assertTrue(np.allclose(expected_topics, topics))