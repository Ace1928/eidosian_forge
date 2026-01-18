import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def _pairwise_average(self, function):
    """
        Calculates the average of function results for each coder pair
        """
    total = 0
    n = 0
    s = self.C.copy()
    for cA in self.C:
        s.remove(cA)
        for cB in s:
            total += function(cA, cB)
            n += 1
    ret = total / n
    return ret