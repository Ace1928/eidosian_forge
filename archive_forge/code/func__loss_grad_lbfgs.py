import sys
import time
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.optimize import minimize
from ..base import (
from ..decomposition import PCA
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances
from ..preprocessing import LabelEncoder
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import softmax
from ..utils.multiclass import check_classification_targets
from ..utils.random import check_random_state
from ..utils.validation import check_array, check_is_fitted
def _loss_grad_lbfgs(self, transformation, X, same_class_mask, sign=1.0):
    """Compute the loss and the loss gradient w.r.t. `transformation`.

        Parameters
        ----------
        transformation : ndarray of shape (n_components * n_features,)
            The raveled linear transformation on which to compute loss and
            evaluate gradient.

        X : ndarray of shape (n_samples, n_features)
            The training samples.

        same_class_mask : ndarray of shape (n_samples, n_samples)
            A mask where `mask[i, j] == 1` if `X[i]` and `X[j]` belong
            to the same class, and `0` otherwise.

        Returns
        -------
        loss : float
            The loss computed for the given transformation.

        gradient : ndarray of shape (n_components * n_features,)
            The new (flattened) gradient of the loss.
        """
    if self.n_iter_ == 0:
        self.n_iter_ += 1
        if self.verbose:
            header_fields = ['Iteration', 'Objective Value', 'Time(s)']
            header_fmt = '{:>10} {:>20} {:>10}'
            header = header_fmt.format(*header_fields)
            cls_name = self.__class__.__name__
            print('[{}]'.format(cls_name))
            print('[{}] {}\n[{}] {}'.format(cls_name, header, cls_name, '-' * len(header)))
    t_funcall = time.time()
    transformation = transformation.reshape(-1, X.shape[1])
    X_embedded = np.dot(X, transformation.T)
    p_ij = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(p_ij, np.inf)
    p_ij = softmax(-p_ij)
    masked_p_ij = p_ij * same_class_mask
    p = np.sum(masked_p_ij, axis=1, keepdims=True)
    loss = np.sum(p)
    weighted_p_ij = masked_p_ij - p_ij * p
    weighted_p_ij_sym = weighted_p_ij + weighted_p_ij.T
    np.fill_diagonal(weighted_p_ij_sym, -weighted_p_ij.sum(axis=0))
    gradient = 2 * X_embedded.T.dot(weighted_p_ij_sym).dot(X)
    if self.verbose:
        t_funcall = time.time() - t_funcall
        values_fmt = '[{}] {:>10} {:>20.6e} {:>10.2f}'
        print(values_fmt.format(self.__class__.__name__, self.n_iter_, loss, t_funcall))
        sys.stdout.flush()
    return (sign * loss, sign * gradient.ravel())