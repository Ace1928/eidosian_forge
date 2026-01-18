import logging
import unittest
import numbers
from os import remove
import numpy as np
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import atmodel
from gensim import matutils
from gensim.test import basetmtests
from gensim.test.utils import (datapath,
from gensim.matutils import jensen_shannon
class TestAuthorTopicModel(unittest.TestCase, basetmtests.TestBaseTopicModel):

    def setUp(self):
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = atmodel.AuthorTopicModel
        self.model = self.class_(corpus, id2word=dictionary, author2doc=author2doc, num_topics=2, passes=100)

    def test_transform(self):
        passed = False
        for i in range(25):
            model = self.class_(id2word=dictionary, num_topics=2, passes=100, random_state=0)
            model.update(corpus, author2doc)
            jill_topics = model.get_author_topics('jill')
            vec = matutils.sparse2full(jill_topics, 2)
            expected = [0.91, 0.08]
            passed = np.allclose(sorted(vec), sorted(expected), atol=0.1)
            if passed:
                break
            logging.warning('Author-topic model failed to converge on attempt %i (got %s, expected %s)', i, sorted(vec), sorted(expected))
        self.assertTrue(passed)

    def test_basic(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)
        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        self.assertTrue(all(jill_topics > 0))

    def test_empty_document(self):
        local_texts = common_texts + [['only_occurs_once_in_corpus_and_alone_in_doc']]
        dictionary = Dictionary(local_texts)
        dictionary.filter_extremes(no_below=2)
        corpus = [dictionary.doc2bow(text) for text in local_texts]
        a2d = author2doc.copy()
        a2d['joaquin'] = [len(local_texts) - 1]
        self.class_(corpus, author2doc=a2d, id2word=dictionary, num_topics=2)

    def test_author2doc_missing(self):
        model = self.class_(corpus, author2doc=author2doc, doc2author=doc2author, id2word=dictionary, num_topics=2, random_state=0)
        model2 = self.class_(corpus, doc2author=doc2author, id2word=dictionary, num_topics=2, random_state=0)
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def test_doc2author_missing(self):
        model = self.class_(corpus, author2doc=author2doc, doc2author=doc2author, id2word=dictionary, num_topics=2, random_state=0)
        model2 = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, random_state=0)
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def test_update(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)
        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        model.update()
        jill_topics2 = model.get_author_topics('jill')
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertFalse(all(np.equal(jill_topics, jill_topics2)))

    def test_update_new_data_old_author(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)
        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        model.update(corpus_new, author2doc_new)
        jill_topics2 = model.get_author_topics('jill')
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertFalse(all(np.equal(jill_topics, jill_topics2)))

    def test_update_new_data_new_author(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2)
        model.update(corpus_new, author2doc_new)
        sally_topics = model.get_author_topics('sally')
        sally_topics = matutils.sparse2full(sally_topics, model.num_topics)
        self.assertTrue(all(sally_topics > 0))

    def test_serialized(self):
        model = self.class_(self.corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, serialized=True, serialization_path=datapath('testcorpus_serialization.mm'))
        jill_topics = model.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        self.assertTrue(all(jill_topics > 0))
        model.update()
        jill_topics2 = model.get_author_topics('jill')
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertFalse(all(np.equal(jill_topics, jill_topics2)))
        model.update(corpus_new, author2doc_new)
        sally_topics = model.get_author_topics('sally')
        sally_topics = matutils.sparse2full(sally_topics, model.num_topics)
        self.assertTrue(all(sally_topics > 0))
        remove(datapath('testcorpus_serialization.mm'))

    def test_transform_serialized(self):
        passed = False
        for i in range(25):
            model = self.class_(id2word=dictionary, num_topics=2, passes=100, random_state=0, serialized=True, serialization_path=datapath('testcorpus_serialization.mm'))
            model.update(self.corpus, author2doc)
            jill_topics = model.get_author_topics('jill')
            vec = matutils.sparse2full(jill_topics, 2)
            expected = [0.91, 0.08]
            passed = np.allclose(sorted(vec), sorted(expected), atol=0.1)
            remove(datapath('testcorpus_serialization.mm'))
            if passed:
                break
            logging.warning('Author-topic model failed to converge on attempt %i (got %s, expected %s)', i, sorted(vec), sorted(expected))
        self.assertTrue(passed)

    def test_alpha_auto(self):
        model1 = self.class_(corpus, author2doc=author2doc, id2word=dictionary, alpha='symmetric', passes=10, num_topics=2)
        modelauto = self.class_(corpus, author2doc=author2doc, id2word=dictionary, alpha='auto', passes=10, num_topics=2)
        self.assertFalse(all(np.equal(model1.alpha, modelauto.alpha)))

    def test_alpha(self):
        kwargs = dict(author2doc=author2doc, id2word=dictionary, num_topics=2, alpha=None)
        expected_shape = (2,)
        self.class_(**kwargs)
        kwargs['alpha'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.5, 0.5])))
        kwargs['alpha'] = 'asymmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(np.allclose(model.alpha, [0.630602, 0.369398]))
        kwargs['alpha'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))
        kwargs['alpha'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([3, 3])))
        kwargs['alpha'] = [0.3, 0.3]
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))
        kwargs['alpha'] = np.array([0.3, 0.3])
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        self.assertTrue(all(model.alpha == np.array([0.3, 0.3])))
        kwargs['alpha'] = [0.3, 0.3, 0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['alpha'] = [[0.3], [0.3]]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['alpha'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['alpha'] = 'gensim is cool'
        self.assertRaises(ValueError, self.class_, **kwargs)

    def test_eta_auto(self):
        model1 = self.class_(corpus, author2doc=author2doc, id2word=dictionary, eta='symmetric', passes=10, num_topics=2)
        modelauto = self.class_(corpus, author2doc=author2doc, id2word=dictionary, eta='auto', passes=10, num_topics=2)
        self.assertFalse(all(np.equal(model1.eta, modelauto.eta)))

    def test_eta(self):
        kwargs = dict(author2doc=author2doc, id2word=dictionary, num_topics=2, eta=None)
        num_terms = len(dictionary)
        expected_shape = (num_terms,)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.5] * num_terms)))
        kwargs['eta'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.5] * num_terms)))
        kwargs['eta'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))
        kwargs['eta'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([3] * num_terms)))
        kwargs['eta'] = [0.3] * num_terms
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))
        kwargs['eta'] = np.array([0.3] * num_terms)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        self.assertTrue(all(model.eta == np.array([0.3] * num_terms)))
        testeta = np.array([[0.5] * len(dictionary)] * 2)
        kwargs['eta'] = testeta
        self.class_(**kwargs)
        kwargs['eta'] = testeta.reshape(tuple(reversed(testeta.shape)))
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['eta'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['eta'] = [0.3] * (num_terms + 1)
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['eta'] = 'gensim is cool'
        self.assertRaises(ValueError, self.class_, **kwargs)
        kwargs['eta'] = 'asymmetric'
        self.assertRaises(ValueError, self.class_, **kwargs)

    def test_top_topics(self):
        top_topics = self.model.top_topics(corpus)
        for topic, score in top_topics:
            self.assertTrue(isinstance(topic, list))
            self.assertTrue(isinstance(score, float))
            for v, k in topic:
                self.assertTrue(isinstance(k, str))
                self.assertTrue(isinstance(v, float))

    def test_get_topic_terms(self):
        topic_terms = self.model.get_topic_terms(1)
        for k, v in topic_terms:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(isinstance(v, float))

    def test_get_author_topics(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
        author_topics = []
        for a in model.id2author.values():
            author_topics.append(model.get_author_topics(a))
        for topic in author_topics:
            self.assertTrue(isinstance(topic, list))
            for k, v in topic:
                self.assertTrue(isinstance(k, int))
                self.assertTrue(isinstance(v, float))

    def test_term_topics(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
        result = model.get_term_topics(2)
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(probability, float))
        result = model.get_term_topics(str(model.id2word[2]))
        for topic_no, probability in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(isinstance(probability, float))

    def test_new_author_topics(self):
        model = self.class_(corpus, author2doc=author2doc, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
        author2doc_newauthor = {}
        author2doc_newauthor['test'] = [0, 1]
        model.update(corpus=corpus[0:2], author2doc=author2doc_newauthor)
        state_gamma_len = len(model.state.gamma)
        author2doc_len = len(model.author2doc)
        author2id_len = len(model.author2id)
        id2author_len = len(model.id2author)
        doc2author_len = len(model.doc2author)
        new_author_topics = model.get_new_author_topics(corpus=corpus[0:2])
        for k, v in new_author_topics:
            self.assertTrue(isinstance(k, int))
            self.assertTrue(isinstance(v, float))
        similarity = 1 / (1 + jensen_shannon(model['test'], new_author_topics))
        self.assertTrue(similarity >= 0.9)
        with self.assertRaises(TypeError):
            model.get_new_author_topics(corpus=corpus[0])
        self.assertEqual(state_gamma_len, len(model.state.gamma))
        self.assertEqual(author2doc_len, len(model.author2doc))
        self.assertEqual(author2id_len, len(model.author2id))
        self.assertEqual(id2author_len, len(model.id2author))
        self.assertEqual(doc2author_len, len(model.doc2author))

    def test_passes(self):
        self.longMessage = True
        test_rhots = []
        model = self.class_(id2word=dictionary, chunksize=1, num_topics=2)

        def final_rhot(model):
            return pow(model.offset + 1 * model.num_updates / model.chunksize, -model.decay)
        for _ in range(5):
            model.update(corpus, author2doc)
            test_rhots.append(final_rhot(model))
        for passes in [1, 5, 10, 50, 100]:
            model = self.class_(id2word=dictionary, chunksize=1, num_topics=2, passes=passes)
            self.assertEqual(final_rhot(model), 1.0)
            for test_rhot in test_rhots:
                model.update(corpus, author2doc)
                msg = '{}, {}, {}'.format(passes, model.num_updates, model.state.numdocs)
                self.assertAlmostEqual(final_rhot(model), test_rhot, msg=msg)
            self.assertEqual(model.state.numdocs, len(corpus) * len(test_rhots))
            self.assertEqual(model.num_updates, len(corpus) * len(test_rhots))

    def test_persistence(self):
        fname = get_tmpfile('gensim_models_atmodel.tst')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        self.assertTrue(np.allclose(model.state.gamma, model2.state.gamma))

    def test_persistence_ignore(self):
        fname = get_tmpfile('gensim_models_atmodel_testPersistenceIgnore.tst')
        model = atmodel.AuthorTopicModel(corpus, author2doc=author2doc, num_topics=2)
        model.save(fname, ignore='id2word')
        model2 = atmodel.AuthorTopicModel.load(fname)
        self.assertTrue(model2.id2word is None)
        model.save(fname, ignore=['id2word'])
        model2 = atmodel.AuthorTopicModel.load(fname)
        self.assertTrue(model2.id2word is None)

    def test_persistence_compressed(self):
        fname = get_tmpfile('gensim_models_atmodel.tst.gz')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def test_large_mmap(self):
        fname = get_tmpfile('gensim_models_atmodel.tst')
        model = self.model
        model.save(fname, sep_limit=0)
        model2 = self.class_.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.expElogbeta, np.memmap))
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        jill_topics = model.get_author_topics('jill')
        jill_topics2 = model2.get_author_topics('jill')
        jill_topics = matutils.sparse2full(jill_topics, model.num_topics)
        jill_topics2 = matutils.sparse2full(jill_topics2, model.num_topics)
        self.assertTrue(np.allclose(jill_topics, jill_topics2))

    def test_large_mmap_compressed(self):
        fname = get_tmpfile('gensim_models_atmodel.tst.gz')
        model = self.model
        model.save(fname, sep_limit=0)
        self.assertRaises(IOError, self.class_.load, fname, mmap='r')

    def test_dtype_backward_compatibility(self):
        atmodel_3_0_1_fname = datapath('atmodel_3_0_1_model')
        expected_topics = [(0, 0.06820084297729673), (1, 0.9317991570227033)]
        model = self.class_.load(atmodel_3_0_1_fname)
        topics = model['jane']
        self.assertTrue(np.allclose(expected_topics, topics))