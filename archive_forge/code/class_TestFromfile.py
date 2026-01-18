from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
class TestFromfile(unittest.TestCase):

    def test_decompressed(self):
        with open(datapath('reproduce.dat'), 'rb') as fin:
            self._run(fin)

    def test_compressed(self):
        with gzip.GzipFile(datapath('reproduce.dat.gz'), 'rb') as fin:
            self._run(fin)

    def _run(self, fin):
        actual = fin.read(len(_BYTES))
        self.assertEqual(_BYTES, actual)
        array = gensim.models._fasttext_bin._fromfile(fin, _ARRAY.dtype, _ARRAY.shape[0])
        logger.error('array: %r', array)
        self.assertTrue(np.allclose(_ARRAY, array))