import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class KneserNeyProbDist(ProbDistI):
    """
    Kneser-Ney estimate of a probability distribution. This is a version of
    back-off that counts how likely an n-gram is provided the n-1-gram had
    been seen in training. Extends the ProbDistI interface, requires a trigram
    FreqDist instance to train on. Optionally, a different from default discount
    value can be specified. The default discount is set to 0.75.

    """

    def __init__(self, freqdist, bins=None, discount=0.75):
        """
        :param freqdist: The trigram frequency distribution upon which to base
            the estimation
        :type freqdist: FreqDist
        :param bins: Included for compatibility with nltk.tag.hmm
        :type bins: int or float
        :param discount: The discount applied when retrieving counts of
            trigrams
        :type discount: float (preferred, but can be set to int)
        """
        if not bins:
            self._bins = freqdist.B()
        else:
            self._bins = bins
        self._D = discount
        self._cache = {}
        self._bigrams = defaultdict(int)
        self._trigrams = freqdist
        self._wordtypes_after = defaultdict(float)
        self._trigrams_contain = defaultdict(float)
        self._wordtypes_before = defaultdict(float)
        for w0, w1, w2 in freqdist:
            self._bigrams[w0, w1] += freqdist[w0, w1, w2]
            self._wordtypes_after[w0, w1] += 1
            self._trigrams_contain[w1] += 1
            self._wordtypes_before[w1, w2] += 1

    def prob(self, trigram):
        if len(trigram) != 3:
            raise ValueError('Expected an iterable with 3 members.')
        trigram = tuple(trigram)
        w0, w1, w2 = trigram
        if trigram in self._cache:
            return self._cache[trigram]
        else:
            if trigram in self._trigrams:
                prob = (self._trigrams[trigram] - self.discount()) / self._bigrams[w0, w1]
            elif (w0, w1) in self._bigrams and (w1, w2) in self._wordtypes_before:
                aftr = self._wordtypes_after[w0, w1]
                bfr = self._wordtypes_before[w1, w2]
                leftover_prob = aftr * self.discount() / self._bigrams[w0, w1]
                beta = bfr / (self._trigrams_contain[w1] - aftr)
                prob = leftover_prob * beta
            else:
                prob = 0.0
            self._cache[trigram] = prob
            return prob

    def discount(self):
        """
        Return the value by which counts are discounted. By default set to 0.75.

        :rtype: float
        """
        return self._D

    def set_discount(self, discount):
        """
        Set the value by which counts are discounted to the value of discount.

        :param discount: the new value to discount counts by
        :type discount: float (preferred, but int possible)
        :rtype: None
        """
        self._D = discount

    def samples(self):
        return self._trigrams.keys()

    def max(self):
        return self._trigrams.max()

    def __repr__(self):
        """
        Return a string representation of this ProbDist

        :rtype: str
        """
        return f'<KneserNeyProbDist based on {self._trigrams.N()} trigrams'