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
def _load_specials(self, *args, **kwargs):
    super(PoincareKeyedVectors, self)._load_specials(*args, **kwargs)
    if not hasattr(self, 'vectors'):
        self.vectors = self.__dict__.pop('syn0')