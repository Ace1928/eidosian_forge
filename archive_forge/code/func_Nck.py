import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def Nck(self, c, k):
    return float(sum((1 for x in self.data if x['coder'] == c and x['labels'] == k)))