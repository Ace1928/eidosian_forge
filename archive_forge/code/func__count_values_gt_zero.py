from operator import methodcaller
from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist
def _count_values_gt_zero(distribution):
    """Count values that are greater than zero in a distribution.

    Assumes distribution is either a mapping with counts as values or
    an instance of `nltk.ConditionalFreqDist`.
    """
    as_count = methodcaller('N') if isinstance(distribution, ConditionalFreqDist) else lambda count: count
    return sum((1 for dist_or_count in distribution.values() if as_count(dist_or_count) > 0))