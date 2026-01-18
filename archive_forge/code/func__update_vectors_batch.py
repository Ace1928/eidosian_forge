import csv
import logging
from numbers import Integral
import sys
import time
from collections import defaultdict, Counter
import numpy as np
from numpy import random as np_random, float32 as REAL
from scipy.stats import spearmanr
from gensim import utils, matutils
from gensim.models.keyedvectors import KeyedVectors
def _update_vectors_batch(self, batch):
    """Update vectors for nodes in the given batch.

        Parameters
        ----------
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch containing computed gradients and node indices of the batch for which updates are to be done.

        """
    grad_u, grad_v = (batch.gradients_u, batch.gradients_v)
    indices_u, indices_v = (batch.indices_u, batch.indices_v)
    batch_size = len(indices_u)
    u_updates = (self.alpha * batch.alpha ** 2 / 4 * grad_u).T
    self._handle_duplicates(u_updates, indices_u)
    self.kv.vectors[indices_u] -= u_updates
    self.kv.vectors[indices_u] = self._clip_vectors(self.kv.vectors[indices_u], self.epsilon)
    v_updates = self.alpha * (batch.beta ** 2)[:, np.newaxis] / 4 * grad_v
    v_updates = v_updates.swapaxes(1, 2).swapaxes(0, 1)
    v_updates = v_updates.reshape(((1 + self.negative) * batch_size, self.size))
    self._handle_duplicates(v_updates, indices_v)
    self.kv.vectors[indices_v] -= v_updates
    self.kv.vectors[indices_v] = self._clip_vectors(self.kv.vectors[indices_v], self.epsilon)