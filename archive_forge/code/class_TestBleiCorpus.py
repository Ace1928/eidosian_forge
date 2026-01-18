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
class TestBleiCorpus(CorpusTestCase):

    def setUp(self):
        self.corpus_class = bleicorpus.BleiCorpus
        self.file_extension = '.blei'

    def test_save_format_for_dtm(self):
        corpus = [[(1, 1.0)], [], [(0, 5.0), (2, 1.0)], []]
        test_file = get_tmpfile('gensim_corpus.tst')
        self.corpus_class.save_corpus(test_file, corpus)
        with open(test_file) as f:
            for line in f:
                tokens = line.split()
                words_len = int(tokens[0])
                if words_len > 0:
                    tokens = tokens[1:]
                else:
                    tokens = []
                self.assertEqual(words_len, len(tokens))
                for token in tokens:
                    word, count = token.split(':')
                    self.assertEqual(count, str(int(count)))