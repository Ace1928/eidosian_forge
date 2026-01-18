from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _sample_probdist(self, probdist, p, samples):
    cum_p = 0
    for sample in samples:
        add_p = probdist.prob(sample)
        if cum_p <= p <= cum_p + add_p:
            return sample
        cum_p += add_p
    raise Exception('Invalid probability distribution - does not sum to one')