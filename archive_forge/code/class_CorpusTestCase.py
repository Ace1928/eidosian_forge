from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
class CorpusTestCase(unittest.TestCase):
    TEST_CORPUS = [[(1, 1.0)], [], [(0, 0.5), (2, 1.0)], []]

    def setUp(self):
        self.corpus_class = None
        self.file_extension = None

    def run(self, result=None):
        if type(self) is not CorpusTestCase:
            super(CorpusTestCase, self).run(result)

    def tearDown(self):
        fname = get_tmpfile('gensim_corpus.tst')
        extensions = ['', '', '.bz2', '.gz', '.index', '.vocab']
        for ext in itertools.permutations(extensions, 2):
            try:
                os.remove(fname + ext[0] + ext[1])
            except OSError:
                pass

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_load(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        docs = list(corpus)
        self.assertEqual(len(docs), 9)

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_len(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        corpus = self.corpus_class(fname)
        self.assertEqual(len(corpus), 9)
        if hasattr(corpus, 'index'):
            corpus.index = None
        self.assertEqual(len(corpus), 9)

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_empty_input(self):
        tmpf = get_tmpfile('gensim_corpus.tst')
        with open(tmpf, 'w') as f:
            f.write('')
        with open(tmpf + '.vocab', 'w') as f:
            f.write('')
        corpus = self.corpus_class(tmpf)
        self.assertEqual(len(corpus), 0)
        docs = list(corpus)
        self.assertEqual(len(docs), 0)

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_save(self):
        corpus = self.TEST_CORPUS
        tmpf = get_tmpfile('gensim_corpus.tst')
        self.corpus_class.save_corpus(tmpf, corpus)
        corpus2 = list(self.corpus_class(tmpf))
        self.assertEqual(corpus, corpus2)

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_serialize(self):
        corpus = self.TEST_CORPUS
        tmpf = get_tmpfile('gensim_corpus.tst')
        self.corpus_class.serialize(tmpf, corpus)
        corpus2 = self.corpus_class(tmpf)
        self.assertEqual(corpus, list(corpus2))
        for i in range(len(corpus)):
            self.assertEqual(corpus[i], corpus2[i])
        if isinstance(corpus, indexedcorpus.IndexedCorpus):
            idx = [1, 3, 5, 7]
            self.assertEqual(corpus[idx], corpus2[idx])

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_serialize_compressed(self):
        corpus = self.TEST_CORPUS
        tmpf = get_tmpfile('gensim_corpus.tst')
        for extension in ['.gz', '.bz2']:
            fname = tmpf + extension
            self.corpus_class.serialize(fname, corpus)
            corpus2 = self.corpus_class(fname)
            self.assertEqual(corpus, list(corpus2))
            for i in range(len(corpus)):
                self.assertEqual(corpus[i], corpus2[i])

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_switch_id2word(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        if hasattr(corpus, 'id2word'):
            firstdoc = next(iter(corpus))
            testdoc = set(((to_unicode(corpus.id2word[x]), y) for x, y in firstdoc))
            self.assertEqual(testdoc, {('computer', 1), ('human', 1), ('interface', 1)})
            d = corpus.id2word
            d[0], d[1] = (d[1], d[0])
            corpus.id2word = d
            firstdoc2 = next(iter(corpus))
            testdoc2 = set(((to_unicode(corpus.id2word[x]), y) for x, y in firstdoc2))
            self.assertEqual(testdoc2, {('computer', 1), ('human', 1), ('interface', 1)})

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_indexing(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        docs = list(corpus)
        for idx, doc in enumerate(docs):
            self.assertEqual(doc, corpus[idx])
            self.assertEqual(doc, corpus[np.int64(idx)])
        self.assertEqual(docs, list(corpus[:]))
        self.assertEqual(docs[0:], list(corpus[0:]))
        self.assertEqual(docs[0:-1], list(corpus[0:-1]))
        self.assertEqual(docs[2:4], list(corpus[2:4]))
        self.assertEqual(docs[::2], list(corpus[::2]))
        self.assertEqual(docs[::-1], list(corpus[::-1]))
        c = corpus[:]
        self.assertEqual(docs, list(c))
        self.assertEqual(docs, list(c))
        self.assertEqual(len(docs), len(corpus))
        self.assertEqual(len(docs), len(corpus[:]))
        self.assertEqual(len(docs[::2]), len(corpus[::2]))

        def _get_slice(corpus, slice_):
            return corpus[slice_]
        self.assertRaises(ValueError, _get_slice, corpus, {1})
        self.assertRaises(ValueError, _get_slice, corpus, 1.0)
        c = corpus[[1, 3, 4]]
        self.assertEqual([d for i, d in enumerate(docs) if i in [1, 3, 4]], list(c))
        self.assertEqual([d for i, d in enumerate(docs) if i in [1, 3, 4]], list(c))
        self.assertEqual(len(corpus[[0, 1, -1]]), 3)
        self.assertEqual(len(corpus[np.asarray([0, 1, -1])]), 3)
        corpus_ = TransformedCorpus(DummyTransformer(), corpus)
        if hasattr(corpus, 'index') and corpus.index is not None:
            self.assertEqual(corpus_[0][0][1], docs[0][0][1] + 1)
            self.assertRaises(ValueError, _get_slice, corpus_, {1})
            transformed_docs = [val + 1 for i, d in enumerate(docs) for _, val in d if i in [1, 3, 4]]
            self.assertEqual(transformed_docs, list((v for doc in corpus_[[1, 3, 4]] for _, v in doc)))
            self.assertEqual(3, len(corpus_[[1, 3, 4]]))
        else:
            self.assertRaises(RuntimeError, _get_slice, corpus_, [1, 3, 4])
            self.assertRaises(RuntimeError, _get_slice, corpus_, {1})
            self.assertRaises(RuntimeError, _get_slice, corpus_, 1.0)