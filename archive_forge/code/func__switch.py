import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _switch(self, r, nr):
    """
        Calculate the r frontier where we must switch from Nr to Sr
        when estimating E[Nr].
        """
    for i, r_ in enumerate(r):
        if len(r) == i + 1 or r[i + 1] != r_ + 1:
            self._switch_at = r_
            break
        Sr = self.smoothedNr
        smooth_r_star = (r_ + 1) * Sr(r_ + 1) / Sr(r_)
        unsmooth_r_star = (r_ + 1) * nr[i + 1] / nr[i]
        std = math.sqrt(self._variance(r_, nr[i], nr[i + 1]))
        if abs(unsmooth_r_star - smooth_r_star) <= 1.96 * std:
            self._switch_at = r_
            break