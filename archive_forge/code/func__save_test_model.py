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
def _save_test_model(out_base_fname, model_params):
    inp_fname = datapath('lee_background.cor')
    model_type = 'cbow' if model_params['sg'] == 0 else 'skipgram'
    size = str(model_params['vector_size'])
    seed = str(model_params['seed'])
    cmd = [FT_CMD, model_type, '-input', inp_fname, '-output', out_base_fname, '-dim', size, '-seed', seed]
    subprocess.check_call(cmd)