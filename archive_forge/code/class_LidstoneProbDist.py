import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class LidstoneProbDist(ProbDistI):
    """
    The Lidstone estimate for the probability distribution of the
    experiment used to generate a frequency distribution.  The
    "Lidstone estimate" is parameterized by a real number *gamma*,
    which typically ranges from 0 to 1.  The Lidstone estimate
    approximates the probability of a sample with count *c* from an
    experiment with *N* outcomes and *B* bins as
    ``c+gamma)/(N+B*gamma)``.  This is equivalent to adding
    *gamma* to the count for each bin, and taking the maximum
    likelihood estimate of the resulting frequency distribution.
    """
    SUM_TO_ONE = False

    def __init__(self, freqdist, gamma, bins=None):
        """
        Use the Lidstone estimate to create a probability distribution
        for the experiment used to generate ``freqdist``.

        :type freqdist: FreqDist
        :param freqdist: The frequency distribution that the
            probability estimates should be based on.
        :type gamma: float
        :param gamma: A real number used to parameterize the
            estimate.  The Lidstone estimate is equivalent to adding
            *gamma* to the count for each bin, and taking the
            maximum likelihood estimate of the resulting frequency
            distribution.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        if bins == 0 or (bins is None and freqdist.N() == 0):
            name = self.__class__.__name__[:-8]
            raise ValueError('A %s probability distribution ' % name + 'must have at least one bin.')
        if bins is not None and bins < freqdist.B():
            name = self.__class__.__name__[:-8]
            raise ValueError('\nThe number of bins in a %s distribution ' % name + '(%d) must be greater than or equal to\n' % bins + 'the number of bins in the FreqDist used ' + 'to create it (%d).' % freqdist.B())
        self._freqdist = freqdist
        self._gamma = float(gamma)
        self._N = self._freqdist.N()
        if bins is None:
            bins = freqdist.B()
        self._bins = bins
        self._divisor = self._N + bins * gamma
        if self._divisor == 0.0:
            self._gamma = 0
            self._divisor = 1

    def freqdist(self):
        """
        Return the frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._freqdist

    def prob(self, sample):
        c = self._freqdist[sample]
        return (c + self._gamma) / self._divisor

    def max(self):
        return self._freqdist.max()

    def samples(self):
        return self._freqdist.keys()

    def discount(self):
        gb = self._gamma * self._bins
        return gb / (self._N + gb)

    def __repr__(self):
        """
        Return a string representation of this ``ProbDist``.

        :rtype: str
        """
        return '<LidstoneProbDist based on %d samples>' % self._freqdist.N()