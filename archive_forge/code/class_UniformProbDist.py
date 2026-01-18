import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class UniformProbDist(ProbDistI):
    """
    A probability distribution that assigns equal probability to each
    sample in a given set; and a zero probability to all other
    samples.
    """

    def __init__(self, samples):
        """
        Construct a new uniform probability distribution, that assigns
        equal probability to each sample in ``samples``.

        :param samples: The samples that should be given uniform
            probability.
        :type samples: list
        :raise ValueError: If ``samples`` is empty.
        """
        if len(samples) == 0:
            raise ValueError('A Uniform probability distribution must ' + 'have at least one sample.')
        self._sampleset = set(samples)
        self._prob = 1.0 / len(self._sampleset)
        self._samples = list(self._sampleset)

    def prob(self, sample):
        return self._prob if sample in self._sampleset else 0

    def max(self):
        return self._samples[0]

    def samples(self):
        return self._samples

    def __repr__(self):
        return '<UniformProbDist with %d samples>' % len(self._sampleset)