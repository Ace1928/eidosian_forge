import itertools
import logging
import multiprocessing as mp
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sps
from gensim import utils
from gensim.models.word2vec import Word2Vec
class PatchedWordOccurrenceAccumulator(WordOccurrenceAccumulator):
    """Monkey patched for multiprocessing worker usage, to move some of the logic to the master process."""

    def _iter_texts(self, texts):
        return texts