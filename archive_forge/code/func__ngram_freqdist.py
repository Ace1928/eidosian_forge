import itertools as _itertools
from nltk.metrics import (
from nltk.metrics.spearman import ranks_from_scores, spearman_correlation
from nltk.probability import FreqDist
from nltk.util import ngrams
@staticmethod
def _ngram_freqdist(words, n):
    return FreqDist((tuple(words[i:i + n]) for i in range(len(words) - 1)))