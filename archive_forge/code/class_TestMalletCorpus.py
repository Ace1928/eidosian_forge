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
class TestMalletCorpus(TestLowCorpus):
    TEST_CORPUS = [[(1, 1)], [], [(0, 2), (2, 1)], []]
    CORPUS_LINE = '#3  lang mom  wash  window window was washed'

    def setUp(self):
        self.corpus_class = malletcorpus.MalletCorpus
        self.file_extension = '.mallet'

    def test_load_with_metadata(self):
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        corpus = self.corpus_class(fname)
        corpus.metadata = True
        self.assertEqual(len(corpus), 9)
        docs = list(corpus)
        self.assertEqual(len(docs), 9)
        for i, docmeta in enumerate(docs):
            doc, metadata = docmeta
            self.assertEqual(metadata[0], str(i + 1))
            self.assertEqual(metadata[1], 'en')

    def test_line2doc(self):
        super(TestMalletCorpus, self).test_line2doc()
        fname = datapath('testcorpus.' + self.file_extension.lstrip('.'))
        id2word = {1: 'mom', 2: 'window'}
        corpus = self.corpus_class(fname, id2word=id2word, metadata=True)
        corpus.use_wordids = False
        doc, (docid, doclang) = corpus.line2doc(self.CORPUS_LINE)
        self.assertEqual(docid, '#3')
        self.assertEqual(doclang, 'lang')
        self.assertEqual(sorted(doc), [('mom', 1), ('was', 1), ('wash', 1), ('washed', 1), ('window', 2)])
        corpus.use_wordids = True
        doc, (docid, doclang) = corpus.line2doc(self.CORPUS_LINE)
        self.assertEqual(docid, '#3')
        self.assertEqual(doclang, 'lang')
        self.assertEqual(sorted(doc), [(1, 1), (2, 2)])