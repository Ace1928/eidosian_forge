from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
class ConcatenatedDocvecs:

    def __init__(self, models):
        self.models = models

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])