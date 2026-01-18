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
class TestMmCorpusWithIndex(CorpusTestCase):

    def setUp(self):
        self.corpus_class = mmcorpus.MmCorpus
        self.corpus = self.corpus_class(datapath('test_mmcorpus_with_index.mm'))
        self.file_extension = '.mm'

    def test_serialize_compressed(self):
        pass

    def test_closed_file_object(self):
        file_obj = open(datapath('testcorpus.mm'))
        f = file_obj.closed
        mmcorpus.MmCorpus(file_obj)
        s = file_obj.closed
        self.assertEqual(f, 0)
        self.assertEqual(s, 0)

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_load(self):
        self.assertEqual(self.corpus.num_docs, 9)
        self.assertEqual(self.corpus.num_terms, 12)
        self.assertEqual(self.corpus.num_nnz, 28)
        it = iter(self.corpus)
        self.assertEqual(next(it), [(0, 1.0), (1, 1.0), (2, 1.0)])
        self.assertEqual(next(it), [(0, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0)])
        self.assertEqual(next(it), [(2, 1.0), (5, 1.0), (7, 1.0), (8, 1.0)])
        self.assertEqual(self.corpus[3], [(1, 1.0), (5, 2.0), (8, 1.0)])
        self.assertEqual(tuple(self.corpus.index), (97, 121, 169, 201, 225, 249, 258, 276, 303))