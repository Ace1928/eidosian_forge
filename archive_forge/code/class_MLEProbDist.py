import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class MLEProbDist(ProbDistI):
    """
    The maximum likelihood estimate for the probability distribution
    of the experiment used to generate a frequency distribution.  The
    "maximum likelihood estimate" approximates the probability of
    each sample as the frequency of that sample in the frequency
    distribution.
    """

    def __init__(self, freqdist, bins=None):
        """
        Use the maximum likelihood estimate to create a probability
        distribution for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        """
        self._freqdist = freqdist

    def freqdist(self):
        """
        Return the frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._freqdist

    def prob(self, sample):
        return self._freqdist.freq(sample)

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def __repr__(self):
        """
        :rtype: str
        :return: A string representation of this ``ProbDist``.
        """
        return '<MLEProbDist based on %d samples>' % self._freqdist.N()