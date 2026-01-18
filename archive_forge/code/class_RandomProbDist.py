import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class RandomProbDist(ProbDistI):
    """
    Generates a random probability distribution whereby each sample
    will be between 0 and 1 with equal probability (uniform random distribution.
    Also called a continuous uniform distribution).
    """

    def __init__(self, samples):
        if len(samples) == 0:
            raise ValueError('A probability distribution must ' + 'have at least one sample.')
        self._probs = self.unirand(samples)
        self._samples = list(self._probs.keys())

    @classmethod
    def unirand(cls, samples):
        """
        The key function that creates a randomized initial distribution
        that still sums to 1. Set as a dictionary of prob values so that
        it can still be passed to MutableProbDist and called with identical
        syntax to UniformProbDist
        """
        samples = set(samples)
        randrow = [random.random() for i in range(len(samples))]
        total = sum(randrow)
        for i, x in enumerate(randrow):
            randrow[i] = x / total
        total = sum(randrow)
        if total != 1:
            randrow[-1] -= total - 1
        return {s: randrow[i] for i, s in enumerate(samples)}

    def max(self):
        if not hasattr(self, '_max'):
            self._max = max(((p, v) for v, p in self._probs.items()))[1]
        return self._max

    def prob(self, sample):
        return self._probs.get(sample, 0)

    def samples(self):
        return self._samples

    def __repr__(self):
        return '<RandomUniformProbDist with %d samples>' % len(self._probs)