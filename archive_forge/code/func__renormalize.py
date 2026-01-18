import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _renormalize(self, r, nr):
    """
        It is necessary to renormalize all the probability estimates to
        ensure a proper probability distribution results. This can be done
        by keeping the estimate of the probability mass for unseen items as
        N(1)/N and renormalizing all the estimates for previously seen items
        (as Gale and Sampson (1995) propose). (See M&S P.213, 1999)
        """
    prob_cov = 0.0
    for r_, nr_ in zip(r, nr):
        prob_cov += nr_ * self._prob_measure(r_)
    if prob_cov:
        self._renormal = (1 - self._prob_measure(0)) / prob_cov