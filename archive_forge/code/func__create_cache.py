from other tagging techniques which often tag each word individually, seeking
import itertools
import re
from nltk.metrics import accuracy
from nltk.probability import (
from nltk.tag.api import TaggerI
from nltk.util import LazyMap, unique_list
def _create_cache(self):
    """
        The cache is a tuple (P, O, X, S) where:

          - S maps symbols to integers.  I.e., it is the inverse
            mapping from self._symbols; for each symbol s in
            self._symbols, the following is true::

              self._symbols[S[s]] == s

          - O is the log output probabilities::

              O[i,k] = log( P(token[t]=sym[k]|tag[t]=state[i]) )

          - X is the log transition probabilities::

              X[i,j] = log( P(tag[t]=state[j]|tag[t-1]=state[i]) )

          - P is the log prior probabilities::

              P[i] = log( P(tag[0]=state[i]) )
        """
    if not self._cache:
        N = len(self._states)
        M = len(self._symbols)
        P = np.zeros(N, np.float32)
        X = np.zeros((N, N), np.float32)
        O = np.zeros((N, M), np.float32)
        for i in range(N):
            si = self._states[i]
            P[i] = self._priors.logprob(si)
            for j in range(N):
                X[i, j] = self._transitions[si].logprob(self._states[j])
            for k in range(M):
                O[i, k] = self._output_logprob(si, self._symbols[k])
        S = {}
        for k in range(M):
            S[self._symbols[k]] = k
        self._cache = (P, O, X, S)