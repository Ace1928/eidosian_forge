from __future__ import unicode_literals
import json
import logging
import os.path
import unittest
import numpy as np
from gensim import utils
from gensim.scripts.segment_wiki import segment_all_articles, segment_and_write_all_articles
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.word2vec2tensor import word2vec2tensor
from gensim.models import KeyedVectors
class TestWord2Vec2Tensor(unittest.TestCase):

    def setUp(self):
        self.datapath = datapath('word2vec_pre_kv_c')
        self.output_folder = get_tmpfile('w2v2t_test')
        self.metadata_file = self.output_folder + '_metadata.tsv'
        self.tensor_file = self.output_folder + '_tensor.tsv'
        self.vector_file = self.output_folder + '_vector.tsv'

    def test_conversion(self):
        word2vec2tensor(word2vec_model_path=self.datapath, tensor_filename=self.output_folder)
        with utils.open(self.metadata_file, 'rb') as f:
            metadata = f.readlines()
        with utils.open(self.tensor_file, 'rb') as f:
            vectors = f.readlines()
        with utils.open(self.datapath, 'rb') as f:
            first_line = f.readline().strip()
        number_words, vector_size = map(int, first_line.split(b' '))
        self.assertTrue(len(metadata) == len(vectors) == number_words, 'Metadata file %s and tensor file %s imply different number of rows.' % (self.metadata_file, self.tensor_file))
        metadata = [word.strip() for word in metadata]
        vectors = [vector.replace(b'\t', b' ') for vector in vectors]
        orig_model = KeyedVectors.load_word2vec_format(self.datapath, binary=False)
        for word, vector in zip(metadata, vectors):
            word_string = word.decode('utf8')
            vector_string = vector.decode('utf8')
            vector_array = np.array(list(map(float, vector_string.split())))
            np.testing.assert_almost_equal(orig_model[word_string], vector_array, decimal=5)