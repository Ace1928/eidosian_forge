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
def _train_model_with_pretrained_vectors():
    """Generate toy-model-pretrained.bin for use in test_load_native_pretrained.

    Requires https://github.com/facebookresearch/fastText/tree/master/python to be installed.

    """
    import fastText
    training_text = datapath('toy-data.txt')
    pretrained_file = datapath('pretrained.vec')
    model = fastText.train_unsupervised(training_text, bucket=100, model='skipgram', dim=5, pretrainedVectors=pretrained_file)
    model.save_model(datapath('toy-model-pretrained.bin'))