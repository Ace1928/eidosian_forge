import logging
from itertools import groupby
from operator import itemgetter
from nltk.internals import deprecated
from nltk.metrics.distance import binary_distance
from nltk.probability import ConditionalFreqDist, FreqDist
def Ao(self, cA, cB):
    """Observed agreement between two coders on all items."""
    data = self._grouped_data('item', (x for x in self.data if x['coder'] in (cA, cB)))
    ret = sum((self.agr(cA, cB, item, item_data) for item, item_data in data)) / len(self.I)
    log.debug('Observed agreement between %s and %s: %f', cA, cB, ret)
    return ret