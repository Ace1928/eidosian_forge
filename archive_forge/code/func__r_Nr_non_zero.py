import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
def _r_Nr_non_zero(self):
    r_Nr = self._freqdist.r_Nr()
    del r_Nr[0]
    return r_Nr