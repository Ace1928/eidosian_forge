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
def _check_gradients(self, relations, all_negatives, batch, tol=1e-08):
    """Compare computed gradients for batch to autograd gradients.

        Parameters
        ----------
        relations : list of tuples
            List of tuples of positive examples of the form (node_1_index, node_2_index).
        all_negatives : list of lists
            List of lists of negative samples for each node_1 in the positive examples.
        batch : :class:`~gensim.models.poincare.PoincareBatch`
            Batch for which computed gradients are to be checked.
        tol : float, optional
            The maximum error between our computed gradients and the reference ones from autograd.

        """
    if not AUTOGRAD_PRESENT:
        logger.warning('autograd could not be imported, cannot do gradient checking')
        logger.warning('please install autograd to enable gradient checking')
        return
    if self._loss_grad is None:
        self._loss_grad = grad(PoincareModel._loss_fn)
    max_diff = 0.0
    for i, (relation, negatives) in enumerate(zip(relations, all_negatives)):
        u, v = relation
        auto_gradients = self._loss_grad(np.vstack((self.kv.vectors[u], self.kv.vectors[[v] + negatives])), self.regularization_coeff)
        computed_gradients = np.vstack((batch.gradients_u[:, i], batch.gradients_v[:, :, i]))
        diff = np.abs(auto_gradients - computed_gradients).max()
        if diff > max_diff:
            max_diff = diff
    logger.info('max difference between computed gradients and autograd gradients: %.10f', max_diff)
    assert max_diff < tol, 'Max difference between computed gradients and autograd gradients %.10f, greater than tolerance %.10f' % (max_diff, tol)