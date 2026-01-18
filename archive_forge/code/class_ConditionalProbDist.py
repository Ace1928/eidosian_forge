import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class ConditionalProbDist(ConditionalProbDistI):
    """
    A conditional probability distribution modeling the experiments
    that were used to generate a conditional frequency distribution.
    A ConditionalProbDist is constructed from a
    ``ConditionalFreqDist`` and a ``ProbDist`` factory:

    - The ``ConditionalFreqDist`` specifies the frequency
      distribution for each condition.
    - The ``ProbDist`` factory is a function that takes a
      condition's frequency distribution, and returns its
      probability distribution.  A ``ProbDist`` class's name (such as
      ``MLEProbDist`` or ``HeldoutProbDist``) can be used to specify
      that class's constructor.

    The first argument to the ``ProbDist`` factory is the frequency
    distribution that it should model; and the remaining arguments are
    specified by the ``factory_args`` parameter to the
    ``ConditionalProbDist`` constructor.  For example, the following
    code constructs a ``ConditionalProbDist``, where the probability
    distribution for each condition is an ``ELEProbDist`` with 10 bins:

        >>> from nltk.corpus import brown
        >>> from nltk.probability import ConditionalFreqDist
        >>> from nltk.probability import ConditionalProbDist, ELEProbDist
        >>> cfdist = ConditionalFreqDist(brown.tagged_words()[:5000])
        >>> cpdist = ConditionalProbDist(cfdist, ELEProbDist, 10)
        >>> cpdist['passed'].max()
        'VBD'
        >>> cpdist['passed'].prob('VBD') #doctest: +ELLIPSIS
        0.423...

    """

    def __init__(self, cfdist, probdist_factory, *factory_args, **factory_kw_args):
        """
        Construct a new conditional probability distribution, based on
        the given conditional frequency distribution and ``ProbDist``
        factory.

        :type cfdist: ConditionalFreqDist
        :param cfdist: The ``ConditionalFreqDist`` specifying the
            frequency distribution for each condition.
        :type probdist_factory: class or function
        :param probdist_factory: The function or class that maps
            a condition's frequency distribution to its probability
            distribution.  The function is called with the frequency
            distribution as its first argument,
            ``factory_args`` as its remaining arguments, and
            ``factory_kw_args`` as keyword arguments.
        :type factory_args: (any)
        :param factory_args: Extra arguments for ``probdist_factory``.
            These arguments are usually used to specify extra
            properties for the probability distributions of individual
            conditions, such as the number of bins they contain.
        :type factory_kw_args: (any)
        :param factory_kw_args: Extra keyword arguments for ``probdist_factory``.
        """
        self._probdist_factory = probdist_factory
        self._factory_args = factory_args
        self._factory_kw_args = factory_kw_args
        for condition in cfdist:
            self[condition] = probdist_factory(cfdist[condition], *factory_args, **factory_kw_args)

    def __missing__(self, key):
        self[key] = self._probdist_factory(FreqDist(), *self._factory_args, **self._factory_kw_args)
        return self[key]