import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class HeldoutProbDist(ProbDistI):
    """
    The heldout estimate for the probability distribution of the
    experiment used to generate two frequency distributions.  These
    two frequency distributions are called the "heldout frequency
    distribution" and the "base frequency distribution."  The
    "heldout estimate" uses uses the "heldout frequency
    distribution" to predict the probability of each sample, given its
    frequency in the "base frequency distribution".

    In particular, the heldout estimate approximates the probability
    for a sample that occurs *r* times in the base distribution as
    the average frequency in the heldout distribution of all samples
    that occur *r* times in the base distribution.

    This average frequency is *Tr[r]/(Nr[r].N)*, where:

    - *Tr[r]* is the total count in the heldout distribution for
      all samples that occur *r* times in the base distribution.
    - *Nr[r]* is the number of samples that occur *r* times in
      the base distribution.
    - *N* is the number of outcomes recorded by the heldout
      frequency distribution.

    In order to increase the efficiency of the ``prob`` member
    function, *Tr[r]/(Nr[r].N)* is precomputed for each value of *r*
    when the ``HeldoutProbDist`` is created.

    :type _estimate: list(float)
    :ivar _estimate: A list mapping from *r*, the number of
        times that a sample occurs in the base distribution, to the
        probability estimate for that sample.  ``_estimate[r]`` is
        calculated by finding the average frequency in the heldout
        distribution of all samples that occur *r* times in the base
        distribution.  In particular, ``_estimate[r]`` =
        *Tr[r]/(Nr[r].N)*.
    :type _max_r: int
    :ivar _max_r: The maximum number of times that any sample occurs
        in the base distribution.  ``_max_r`` is used to decide how
        large ``_estimate`` must be.
    """
    SUM_TO_ONE = False

    def __init__(self, base_fdist, heldout_fdist, bins=None):
        """
        Use the heldout estimate to create a probability distribution
        for the experiment used to generate ``base_fdist`` and
        ``heldout_fdist``.

        :type base_fdist: FreqDist
        :param base_fdist: The base frequency distribution.
        :type heldout_fdist: FreqDist
        :param heldout_fdist: The heldout frequency distribution.
        :type bins: int
        :param bins: The number of sample values that can be generated
            by the experiment that is described by the probability
            distribution.  This value must be correctly set for the
            probabilities of the sample values to sum to one.  If
            ``bins`` is not specified, it defaults to ``freqdist.B()``.
        """
        self._base_fdist = base_fdist
        self._heldout_fdist = heldout_fdist
        self._max_r = base_fdist[base_fdist.max()]
        Tr = self._calculate_Tr()
        r_Nr = base_fdist.r_Nr(bins)
        Nr = [r_Nr[r] for r in range(self._max_r + 1)]
        N = heldout_fdist.N()
        self._estimate = self._calculate_estimate(Tr, Nr, N)

    def _calculate_Tr(self):
        """
        Return the list *Tr*, where *Tr[r]* is the total count in
        ``heldout_fdist`` for all samples that occur *r*
        times in ``base_fdist``.

        :rtype: list(float)
        """
        Tr = [0.0] * (self._max_r + 1)
        for sample in self._heldout_fdist:
            r = self._base_fdist[sample]
            Tr[r] += self._heldout_fdist[sample]
        return Tr

    def _calculate_estimate(self, Tr, Nr, N):
        """
        Return the list *estimate*, where *estimate[r]* is the probability
        estimate for any sample that occurs *r* times in the base frequency
        distribution.  In particular, *estimate[r]* is *Tr[r]/(N[r].N)*.
        In the special case that *N[r]=0*, *estimate[r]* will never be used;
        so we define *estimate[r]=None* for those cases.

        :rtype: list(float)
        :type Tr: list(float)
        :param Tr: the list *Tr*, where *Tr[r]* is the total count in
            the heldout distribution for all samples that occur *r*
            times in base distribution.
        :type Nr: list(float)
        :param Nr: The list *Nr*, where *Nr[r]* is the number of
            samples that occur *r* times in the base distribution.
        :type N: int
        :param N: The total number of outcomes recorded by the heldout
            frequency distribution.
        """
        estimate = []
        for r in range(self._max_r + 1):
            if Nr[r] == 0:
                estimate.append(None)
            else:
                estimate.append(Tr[r] / (Nr[r] * N))
        return estimate

    def base_fdist(self):
        """
        Return the base frequency distribution that this probability
        distribution is based on.

        :rtype: FreqDist
        """
        return self._base_fdist

    def heldout_fdist(self):
        """
        Return the heldout frequency distribution that this
        probability distribution is based on.

        :rtype: FreqDist
        """
        return self._heldout_fdist

    def samples(self):
        return self._base_fdist.keys()

    def prob(self, sample):
        r = self._base_fdist[sample]
        return self._estimate[r]

    def max(self):
        return self._base_fdist.max()

    def discount(self):
        raise NotImplementedError()

    def __repr__(self):
        """
        :rtype: str
        :return: A string representation of this ``ProbDist``.
        """
        s = '<HeldoutProbDist: %d base samples; %d heldout samples>'
        return s % (self._base_fdist.N(), self._heldout_fdist.N())