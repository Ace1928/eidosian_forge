import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _calculate_estimate(self, Tr, Nr, N):
    """
        Return the list *estimate*, where *estimate[r]* is the probability
        estimate for any sample that occurs *r* times in the base frequency
        distribution.  In particular, *estimate[r]* is *Tr[r]/(N[r].N)*.
        In the special case that *N[r]=0*, *estimate[r]* will never be used;
        so we define *estimate[r]=None* for those cases.

        :rtype: list(float)
        :type Tr: list(float)
        :param Tr: the list *Tr*, where *Tr[r]* is the total count in
            the heldout distribution for all samples that occur *r*
            times in base distribution.
        :type Nr: list(float)
        :param Nr: The list *Nr*, where *Nr[r]* is the number of
            samples that occur *r* times in the base distribution.
        :type N: int
        :param N: The total number of outcomes recorded by the heldout
            frequency distribution.
        """
    estimate = []
    for r in range(self._max_r + 1):
        if Nr[r] == 0:
            estimate.append(None)
        else:
            estimate.append(Tr[r] / (Nr[r] * N))
    return estimate