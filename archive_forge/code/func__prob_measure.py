import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _prob_measure(self, count):
    if count == 0 and self._freqdist.N() == 0:
        return 1.0
    elif count == 0 and self._freqdist.N() != 0:
        return self._freqdist.Nr(1) / self._freqdist.N()
    if self._switch_at > count:
        Er_1 = self._freqdist.Nr(count + 1)
        Er = self._freqdist.Nr(count)
    else:
        Er_1 = self.smoothedNr(count + 1)
        Er = self.smoothedNr(count)
    r_star = (count + 1) * Er_1 / Er
    return r_star / self._freqdist.N()