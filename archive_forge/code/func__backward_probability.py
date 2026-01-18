from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _backward_probability(self, unlabeled_sequence):
    """
        Return the backward probability matrix, a T by N array of
        log-probabilities, where T is the length of the sequence and N is the
        number of states. Each entry (t, s) gives the probability of being in
        state s at time t after observing the partial symbol sequence from t
        .. T.

        :return: the backward log probability matrix
        :rtype:  array
        :param unlabeled_sequence: the sequence of unlabeled symbols
        :type unlabeled_sequence: list
        """
    T = len(unlabeled_sequence)
    N = len(self._states)
    beta = _ninf_array((T, N))
    transitions_logprob = self._transitions_matrix().T
    beta[T - 1, :] = np.log2(1)
    for t in range(T - 2, -1, -1):
        symbol = unlabeled_sequence[t + 1][_TEXT]
        outputs = self._outputs_vector(symbol)
        for i in range(N):
            summand = transitions_logprob[i] + beta[t + 1] + outputs
            beta[t, i] = logsumexp2(summand)
    return beta