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
class TestTextCorpus(CorpusTestCase):

    def setUp(self):
        self.corpus_class = textcorpus.TextCorpus
        self.file_extension = '.txt'

    def test_load_with_metadata(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        corpus.metadata = True
        self.assertEqual(len(corpus), 9)
        docs = list(corpus)
        self.assertEqual(len(docs), 9)
        for i, docmeta in enumerate(docs):
            doc, metadata = docmeta
            self.assertEqual(metadata[0], i)

    def test_default_preprocessing(self):
        lines = ['Šéf chomutovských komunistů dostal poštou bílý prášek', 'this is a test for stopwords', 'zf tooth   spaces   ']
        expected = [['Sef', 'chomutovskych', 'komunistu', 'dostal', 'postou', 'bily', 'prasek'], ['test', 'stopwords'], ['tooth', 'spaces']]
        corpus = self.corpus_from_lines(lines)
        texts = list(corpus.get_texts())
        self.assertEqual(expected, texts)

    def corpus_from_lines(self, lines):
        fpath = tempfile.mktemp()
        with codecs.open(fpath, 'w', encoding='utf8') as f:
            f.write('\n'.join(lines))
        return self.corpus_class(fpath)

    def test_sample_text(self):
        lines = ['document%d' % i for i in range(10)]
        corpus = self.corpus_from_lines(lines)
        corpus.tokenizer = lambda text: text.split()
        docs = [doc for doc in corpus.get_texts()]
        sample1 = list(corpus.sample_texts(1))
        self.assertEqual(len(sample1), 1)
        self.assertIn(sample1[0], docs)
        sample2 = list(corpus.sample_texts(len(lines)))
        self.assertEqual(len(sample2), len(corpus))
        for i in range(len(corpus)):
            self.assertEqual(sample2[i], ['document%s' % i])
        with self.assertRaises(ValueError):
            list(corpus.sample_texts(len(corpus) + 1))
        with self.assertRaises(ValueError):
            list(corpus.sample_texts(-1))

    def test_sample_text_length(self):
        lines = ['document%d' % i for i in range(10)]
        corpus = self.corpus_from_lines(lines)
        corpus.tokenizer = lambda text: text.split()
        sample1 = list(corpus.sample_texts(1, length=1))
        self.assertEqual(sample1[0], ['document0'])
        sample2 = list(corpus.sample_texts(2, length=2))
        self.assertEqual(sample2[0], ['document0'])
        self.assertEqual(sample2[1], ['document1'])

    def test_sample_text_seed(self):
        lines = ['document%d' % i for i in range(10)]
        corpus = self.corpus_from_lines(lines)
        sample1 = list(corpus.sample_texts(5, seed=42))
        sample2 = list(corpus.sample_texts(5, seed=42))
        self.assertEqual(sample1, sample2)

    def test_save(self):
        pass

    def test_serialize(self):
        pass

    def test_serialize_compressed(self):
        pass

    def test_indexing(self):
        pass