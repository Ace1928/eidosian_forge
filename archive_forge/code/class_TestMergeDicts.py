import logging
import unittest
import numpy as np
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
class TestMergeDicts(unittest.TestCase):

    def test_merge_dicts(self):
        d1 = {'word1': 5, 'word2': 1, 'word3': 2}
        d2 = {'word1': 2, 'word3': 3, 'word4': 10}
        res_dict = utils.merge_counts(d1, d2)
        expected_dict = {'word1': 7, 'word2': 1, 'word3': 5, 'word4': 10}
        self.assertEqual(res_dict, expected_dict)