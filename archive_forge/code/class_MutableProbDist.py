import array
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter, defaultdict
from functools import reduce
from nltk.internals import raise_unorderable_types
class MutableProbDist(ProbDistI):
    """
    An mutable probdist where the probabilities may be easily modified. This
    simply copies an existing probdist, storing the probability values in a
    mutable dictionary and providing an update method.
    """

    def __init__(self, prob_dist, samples, store_logs=True):
        """
        Creates the mutable probdist based on the given prob_dist and using
        the list of samples given. These values are stored as log
        probabilities if the store_logs flag is set.

        :param prob_dist: the distribution from which to garner the
            probabilities
        :type prob_dist: ProbDist
        :param samples: the complete set of samples
        :type samples: sequence of any
        :param store_logs: whether to store the probabilities as logarithms
        :type store_logs: bool
        """
        self._samples = samples
        self._sample_dict = {samples[i]: i for i in range(len(samples))}
        self._data = array.array('d', [0.0]) * len(samples)
        for i in range(len(samples)):
            if store_logs:
                self._data[i] = prob_dist.logprob(samples[i])
            else:
                self._data[i] = prob_dist.prob(samples[i])
        self._logs = store_logs

    def max(self):
        return max(((p, v) for v, p in self._sample_dict.items()))[1]

    def samples(self):
        return self._samples

    def prob(self, sample):
        i = self._sample_dict.get(sample)
        if i is None:
            return 0.0
        return 2 ** self._data[i] if self._logs else self._data[i]

    def logprob(self, sample):
        i = self._sample_dict.get(sample)
        if i is None:
            return float('-inf')
        return self._data[i] if self._logs else math.log(self._data[i], 2)

    def update(self, sample, prob, log=True):
        """
        Update the probability for the given sample. This may cause the object
        to stop being the valid probability distribution - the user must
        ensure that they update the sample probabilities such that all samples
        have probabilities between 0 and 1 and that all probabilities sum to
        one.

        :param sample: the sample for which to update the probability
        :type sample: any
        :param prob: the new probability
        :type prob: float
        :param log: is the probability already logged
        :type log: bool
        """
        i = self._sample_dict.get(sample)
        assert i is not None
        if self._logs:
            self._data[i] = prob if log else math.log(prob, 2)
        else:
            self._data[i] = 2 ** prob if log else prob