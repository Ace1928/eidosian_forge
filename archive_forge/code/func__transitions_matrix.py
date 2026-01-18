from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _transitions_matrix(self):
    """Return a matrix of transition log probabilities."""
    trans_iter = (self._transitions[sj].logprob(si) for sj in self._states for si in self._states)
    transitions_logprob = np.fromiter(trans_iter, dtype=np.float64)
    N = len(self._states)
    return transitions_logprob.reshape((N, N)).T