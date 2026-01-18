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
def _read_wordvectors_using_fasttext(fasttext_fname, words):

    def line_to_array(line):
        return np.array([float(s) for s in line.split()[1:]], dtype=np.float32)
    cmd = [FT_CMD, 'print-word-vectors', fasttext_fname]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    words_str = '\n'.join(words)
    out, _ = process.communicate(input=words_str.encode('utf-8'))
    return np.array([line_to_array(line) for line in out.splitlines()], dtype=np.float32)