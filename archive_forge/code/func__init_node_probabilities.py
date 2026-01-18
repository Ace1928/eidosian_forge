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
def _init_node_probabilities(self):
    """Initialize a-priori probabilities."""
    counts = self.kv.expandos['count'].astype(np.float64)
    self._node_counts_cumsum = np.cumsum(counts)
    self._node_probabilities = counts / counts.sum()