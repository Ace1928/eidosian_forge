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
class ConcatenatedDoc2Vec:
    """
    Concatenation of multiple models for reproducing the Paragraph Vectors paper.
    Models must have exactly-matching vocabulary and document IDs. (Models should
    be trained separately; this wrapper just returns concatenated results.)
    """

    def __init__(self, models):
        self.models = models
        if hasattr(models[0], 'dv'):
            self.dv = ConcatenatedDocvecs([model.dv for model in models])

    def __getitem__(self, token):
        return np.concatenate([model[token] for model in self.models])

    def __str__(self):
        """Abbreviated name, built from submodels' names"""
        return '+'.join((str(model) for model in self.models))

    @property
    def epochs(self):
        return self.models[0].epochs

    def infer_vector(self, document, alpha=None, min_alpha=None, epochs=None):
        return np.concatenate([model.infer_vector(document, alpha, min_alpha, epochs) for model in self.models])

    def train(self, *ignore_args, **ignore_kwargs):
        pass