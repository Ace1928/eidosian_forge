import logging
import unittest
import os
import bz2
import sys
import tempfile
import subprocess
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import word2vec, keyedvectors
from gensim.utils import check_output
from gensim.test.utils import (
class TestWord2VecSentenceIterators(unittest.TestCase):

    def test_line_sentence_works_with_filename(self):
        """Does LineSentence work with a filename argument?"""
        with utils.open(datapath('lee_background.cor'), 'rb') as orig:
            sentences = word2vec.LineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def test_cython_line_sentence_works_with_filename(self):
        """Does CythonLineSentence work with a filename argument?"""
        from gensim.models import word2vec_corpusfile
        with utils.open(datapath('lee_background.cor'), 'rb') as orig:
            sentences = word2vec_corpusfile.CythonLineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, orig.readline().split())

    def test_line_sentence_works_with_compressed_file(self):
        """Does LineSentence work with a compressed file object argument?"""
        with utils.open(datapath('head500.noblanks.cor'), 'rb') as orig:
            sentences = word2vec.LineSentence(bz2.BZ2File(datapath('head500.noblanks.cor.bz2')))
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def test_line_sentence_works_with_normal_file(self):
        """Does LineSentence work with a file object argument, rather than filename?"""
        with utils.open(datapath('head500.noblanks.cor'), 'rb') as orig:
            with utils.open(datapath('head500.noblanks.cor'), 'rb') as fin:
                sentences = word2vec.LineSentence(fin)
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig.readline()).split())

    def test_path_line_sentences(self):
        """Does PathLineSentences work with a path argument?"""
        with utils.open(os.path.join(datapath('PathLineSentences'), '1.txt'), 'rb') as orig1:
            with utils.open(os.path.join(datapath('PathLineSentences'), '2.txt.bz2'), 'rb') as orig2:
                sentences = word2vec.PathLineSentences(datapath('PathLineSentences'))
                orig = orig1.readlines() + orig2.readlines()
                orig_counter = 0
                for words in sentences:
                    self.assertEqual(words, utils.to_unicode(orig[orig_counter]).split())
                    orig_counter += 1

    def test_path_line_sentences_one_file(self):
        """Does PathLineSentences work with a single file argument?"""
        test_file = os.path.join(datapath('PathLineSentences'), '1.txt')
        with utils.open(test_file, 'rb') as orig:
            sentences = word2vec.PathLineSentences(test_file)
            for words in sentences:
                self.assertEqual(words, utils.to_unicode(orig.readline()).split())