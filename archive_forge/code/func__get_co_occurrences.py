import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
def _get_co_occurrences(self, word_id1, word_id2):
    return self._co_occurrences[word_id1, word_id2]