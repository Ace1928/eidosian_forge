import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _calculate_Tr(self):
    """
        Return the list *Tr*, where *Tr[r]* is the total count in
        ``heldout_fdist`` for all samples that occur *r*
        times in ``base_fdist``.

        :rtype: list(float)
        """
    Tr = [0.0] * (self._max_r + 1)
    for sample in self._heldout_fdist:
        r = self._base_fdist[sample]
        Tr[r] += self._heldout_fdist[sample]
    return Tr