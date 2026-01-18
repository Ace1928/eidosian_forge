import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class ProbDistI(metaclass=ABCMeta):
    """
    A probability distribution for the outcomes of an experiment.  A
    probability distribution specifies how likely it is that an
    experiment will have any given outcome.  For example, a
    probability distribution could be used to predict the probability
    that a token in a document will have a given type.  Formally, a
    probability distribution can be defined as a function mapping from
    samples to nonnegative real numbers, such that the sum of every
    number in the function's range is 1.0.  A ``ProbDist`` is often
    used to model the probability distribution of the experiment used
    to generate a frequency distribution.
    """
    SUM_TO_ONE = True
    'True if the probabilities of the samples in this probability\n       distribution will always sum to one.'

    @abstractmethod
    def __init__(self):
        """
        Classes inheriting from ProbDistI should implement __init__.
        """

    @abstractmethod
    def prob(self, sample):
        """
        Return the probability for a given sample.  Probabilities
        are always real numbers in the range [0, 1].

        :param sample: The sample whose probability
               should be returned.
        :type sample: any
        :rtype: float
        """

    def logprob(self, sample):
        """
        Return the base 2 logarithm of the probability for a given sample.

        :param sample: The sample whose probability
               should be returned.
        :type sample: any
        :rtype: float
        """
        p = self.prob(sample)
        return math.log(p, 2) if p != 0 else _NINF

    @abstractmethod
    def max(self):
        """
        Return the sample with the greatest probability.  If two or
        more samples have the same probability, return one of them;
        which sample is returned is undefined.

        :rtype: any
        """

    @abstractmethod
    def samples(self):
        """
        Return a list of all samples that have nonzero probabilities.
        Use ``prob`` to find the probability of each sample.

        :rtype: list
        """

    def discount(self):
        """
        Return the ratio by which counts are discounted on average: c*/c

        :rtype: float
        """
        return 0.0

    def generate(self):
        """
        Return a randomly selected sample from this probability distribution.
        The probability of returning each sample ``samp`` is equal to
        ``self.prob(samp)``.
        """
        p = random.random()
        p_init = p
        for sample in self.samples():
            p -= self.prob(sample)
            if p <= 0:
                return sample
        if p < 0.0001:
            return sample
        if self.SUM_TO_ONE:
            warnings.warn('Probability distribution %r sums to %r; generate() is returning an arbitrary sample.' % (self, p_init - p))
        return random.choice(list(self.samples()))