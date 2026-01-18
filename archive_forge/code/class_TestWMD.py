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
class TestWMD(unittest.TestCase):

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_nonzero(self):
        """Test basic functionality with a test sentence."""
        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        sentence1 = ['human', 'interface', 'computer']
        sentence2 = ['survey', 'user', 'computer', 'system', 'response', 'time']
        distance = model.wv.wmdistance(sentence1, sentence2)
        self.assertFalse(distance == 0.0)

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_symmetry(self):
        """Check that distance is symmetric."""
        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        sentence1 = ['human', 'interface', 'computer']
        sentence2 = ['survey', 'user', 'computer', 'system', 'response', 'time']
        distance1 = model.wv.wmdistance(sentence1, sentence2)
        distance2 = model.wv.wmdistance(sentence2, sentence1)
        self.assertTrue(np.allclose(distance1, distance2))

    @unittest.skipIf(POT_EXT is False, 'POT not installed')
    def test_identical_sentences(self):
        """Check that the distance from a sentence to itself is zero."""
        model = word2vec.Word2Vec(sentences, min_count=1)
        sentence = ['survey', 'user', 'computer', 'system', 'response', 'time']
        distance = model.wv.wmdistance(sentence, sentence)
        self.assertEqual(0.0, distance)