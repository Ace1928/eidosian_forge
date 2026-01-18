import collections.abc
import logging
import numpy as np
import scipy.sparse
from scipy.stats import halfnorm
from gensim import interfaces
from gensim import matutils
from gensim import utils
from gensim.interfaces import TransformedCorpus
from gensim.models import basemodel, CoherenceModel
from gensim.models.nmf_pgd import solve_h
def _solve_w(self):
    """Update W."""

    def error(WA):
        """An optimized version of 0.5 * trace(WtWA) - trace(WtB)."""
        return 0.5 * np.einsum('ij,ij', WA, self._W) - np.einsum('ij,ij', self._W, self.B)
    eta = self._kappa / np.linalg.norm(self.A)
    for iter_number in range(self._w_max_iter):
        logger.debug('w_error: %s', self._w_error)
        WA = self._W.dot(self.A)
        self._W -= eta * (WA - self.B)
        self._transform()
        error_ = error(WA)
        if self._w_error < np.inf and np.abs((error_ - self._w_error) / self._w_error) < self._w_stop_condition:
            self._w_error = error_
            break
        self._w_error = error_