import warnings
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics import euclidean_distances, pairwise_distances_argmin
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _affinity_propagation(S, *, preference, convergence_iter, max_iter, damping, verbose, return_n_iter, random_state):
    """Main affinity propagation algorithm."""
    n_samples = S.shape[0]
    if n_samples == 1 or _equal_similarities_and_preferences(S, preference):
        warnings.warn('All samples have mutually equal similarities. Returning arbitrary cluster center(s).')
        if preference.flat[0] > S.flat[n_samples - 1]:
            return (np.arange(n_samples), np.arange(n_samples), 0) if return_n_iter else (np.arange(n_samples), np.arange(n_samples))
        else:
            return (np.array([0]), np.array([0] * n_samples), 0) if return_n_iter else (np.array([0]), np.array([0] * n_samples))
    S.flat[::n_samples + 1] = preference
    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))
    tmp = np.zeros((n_samples, n_samples))
    S += (np.finfo(S.dtype).eps * S + np.finfo(S.dtype).tiny * 100) * random_state.standard_normal(size=(n_samples, n_samples))
    e = np.zeros((n_samples, convergence_iter))
    ind = np.arange(n_samples)
    for it in range(max_iter):
        np.add(A, S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)
        np.subtract(S, Y[:, None], tmp)
        tmp[ind, I] = S[ind, I] - Y2
        tmp *= 1 - damping
        R *= damping
        R += tmp
        np.maximum(R, 0, tmp)
        tmp.flat[::n_samples + 1] = R.flat[::n_samples + 1]
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[::n_samples + 1] = dA
        tmp *= 1 - damping
        A *= damping
        A -= tmp
        E = np.diag(A) + np.diag(R) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)
        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = np.sum((se == convergence_iter) + (se == 0)) != n_samples
            if not unconverged and K > 0 or it == max_iter:
                never_converged = False
                if verbose:
                    print('Converged after %d iterations.' % it)
                break
    else:
        never_converged = True
        if verbose:
            print('Did not converge')
    I = np.flatnonzero(E)
    K = I.size
    if K > 0:
        if never_converged:
            warnings.warn('Affinity propagation did not converge, this model may return degenerate cluster centers and labels.', ConvergenceWarning)
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        warnings.warn('Affinity propagation did not converge and this model will not have any cluster centers.', ConvergenceWarning)
        labels = np.array([-1] * n_samples)
        cluster_centers_indices = []
    if return_n_iter:
        return (cluster_centers_indices, labels, it + 1)
    else:
        return (cluster_centers_indices, labels)