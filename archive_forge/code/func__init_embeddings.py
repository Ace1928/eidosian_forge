import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def _init_embeddings(self):
    """Randomly initialize vectors for the items in the vocab."""
    shape = (len(self.kv.index_to_key), self.size)
    self.kv.vectors = self._np_random.uniform(self.init_range[0], self.init_range[1], shape).astype(self.dtype)