import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class DictionaryConditionalProbDist(ConditionalProbDistI):
    """
    An alternative ConditionalProbDist that simply wraps a dictionary of
    ProbDists rather than creating these from FreqDists.
    """

    def __init__(self, probdist_dict):
        """
        :param probdist_dict: a dictionary containing the probdists indexed
            by the conditions
        :type probdist_dict: dict any -> probdist
        """
        self.update(probdist_dict)

    def __missing__(self, key):
        self[key] = DictionaryProbDist()
        return self[key]