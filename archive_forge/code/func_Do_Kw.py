import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def Do_Kw(self, max_distance=1.0):
    """Averaged over all labelers"""
    ret = self._pairwise_average(lambda cA, cB: self.Do_Kw_pairwise(cA, cB, max_distance))
    log.debug('Observed disagreement: %f', ret)
    return ret