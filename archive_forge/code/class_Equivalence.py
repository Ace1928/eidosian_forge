from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
class Equivalence(CovStruct):
    """
    A covariance structure defined in terms of equivalence classes.

    An 'equivalence class' is a set of pairs of observations such that
    the covariance of every pair within the equivalence class has a
    common value.

    Parameters
    ----------
    pairs : dict-like
      A dictionary of dictionaries, where `pairs[group][label]`
      provides the indices of all pairs of observations in the group
      that have the same covariance value.  Specifically,
      `pairs[group][label]` is a tuple `(j1, j2)`, where `j1` and `j2`
      are integer arrays of the same length.  `j1[i], j2[i]` is one
      index pair that belongs to the `label` equivalence class.  Only
      one triangle of each covariance matrix should be included.
      Positions where j1 and j2 have the same value are variance
      parameters.
    labels : array_like
      An array of labels such that every distinct pair of labels
      defines an equivalence class.  Either `labels` or `pairs` must
      be provided.  When the two labels in a pair are equal two
      equivalence classes are defined: one for the diagonal elements
      (corresponding to variances) and one for the off-diagonal
      elements (corresponding to covariances).
    return_cov : bool
      If True, `covariance_matrix` returns an estimate of the
      covariance matrix, otherwise returns an estimate of the
      correlation matrix.

    Notes
    -----
    Using `labels` to define the class is much easier than using
    `pairs`, but is less general.

    Any pair of values not contained in `pairs` will be assigned zero
    covariance.

    The index values in `pairs` are row indices into the `exog`
    matrix.  They are not updated if missing data are present.  When
    using this covariance structure, missing data should be removed
    before constructing the model.

    If using `labels`, after a model is defined using the covariance
    structure it is possible to remove a label pair from the second
    level of the `pairs` dictionary to force the corresponding
    covariance to be zero.

    Examples
    --------
    The following sets up the `pairs` dictionary for a model with two
    groups, equal variance for all observations, and constant
    covariance for all pairs of observations within each group.

    >> pairs = {0: {}, 1: {}}
    >> pairs[0][0] = (np.r_[0, 1, 2], np.r_[0, 1, 2])
    >> pairs[0][1] = np.tril_indices(3, -1)
    >> pairs[1][0] = (np.r_[3, 4, 5], np.r_[3, 4, 5])
    >> pairs[1][2] = 3 + np.tril_indices(3, -1)
    """

    def __init__(self, pairs=None, labels=None, return_cov=False):
        super().__init__()
        if pairs is None and labels is None:
            raise ValueError('Equivalence cov_struct requires either `pairs` or `labels`')
        if pairs is not None and labels is not None:
            raise ValueError('Equivalence cov_struct accepts only one of `pairs` and `labels`')
        if pairs is not None:
            import copy
            self.pairs = copy.deepcopy(pairs)
        if labels is not None:
            self.labels = np.asarray(labels)
        self.return_cov = return_cov

    def _make_pairs(self, i, j):
        """
        Create arrays containing all unique ordered pairs of i, j.

        The arrays i and j must be one-dimensional containing non-negative
        integers.
        """
        mat = np.zeros((len(i) * len(j), 2), dtype=np.int32)
        f = np.ones(len(j))
        mat[:, 0] = np.kron(f, i).astype(np.int32)
        f = np.ones(len(i))
        mat[:, 1] = np.kron(j, f).astype(np.int32)
        mat.sort(1)
        try:
            dtype = np.dtype((np.void, mat.dtype.itemsize * mat.shape[1]))
            bmat = np.ascontiguousarray(mat).view(dtype)
            _, idx = np.unique(bmat, return_index=True)
        except TypeError:
            rs = np.random.RandomState(4234)
            bmat = np.dot(mat, rs.uniform(size=mat.shape[1]))
            _, idx = np.unique(bmat, return_index=True)
        mat = mat[idx, :]
        return (mat[:, 0], mat[:, 1])

    def _pairs_from_labels(self):
        from collections import defaultdict
        pairs = defaultdict(lambda: defaultdict(lambda: None))
        model = self.model
        df = pd.DataFrame({'labels': self.labels, 'groups': model.groups})
        gb = df.groupby(['groups', 'labels'])
        ulabels = np.unique(self.labels)
        for g_ix, g_lb in enumerate(model.group_labels):
            for lx1 in range(len(ulabels)):
                for lx2 in range(lx1 + 1):
                    lb1 = ulabels[lx1]
                    lb2 = ulabels[lx2]
                    try:
                        i1 = gb.groups[g_lb, lb1]
                        i2 = gb.groups[g_lb, lb2]
                    except KeyError:
                        continue
                    i1, i2 = self._make_pairs(i1, i2)
                    clabel = str(lb1) + '/' + str(lb2)
                    jj = np.flatnonzero(i1 == i2)
                    if len(jj) > 0:
                        clabelv = clabel + '/v'
                        pairs[g_lb][clabelv] = (i1[jj], i2[jj])
                    jj = np.flatnonzero(i1 != i2)
                    if len(jj) > 0:
                        i1 = i1[jj]
                        i2 = i2[jj]
                        pairs[g_lb][clabel] = (i1, i2)
        self.pairs = pairs

    def initialize(self, model):
        super().initialize(model)
        if self.model.weights is not None:
            warnings.warn('weights not implemented for equalence cov_struct, using unweighted covariance estimate', NotImplementedWarning)
        if not hasattr(self, 'pairs'):
            self._pairs_from_labels()
        self.dep_params = defaultdict(float)
        self._var_classes = set()
        for gp in self.model.group_labels:
            for lb in self.pairs[gp]:
                j1, j2 = self.pairs[gp][lb]
                if np.any(j1 == j2):
                    if not np.all(j1 == j2):
                        warnings.warn('equivalence class contains both variance and covariance parameters', OutputWarning)
                    self._var_classes.add(lb)
                    self.dep_params[lb] = 1
        rx = -1 * np.ones(len(self.model.endog), dtype=np.int32)
        for g_ix, g_lb in enumerate(self.model.group_labels):
            ii = self.model.group_indices[g_lb]
            rx[ii] = np.arange(len(ii), dtype=np.int32)
        for gp in self.model.group_labels:
            for lb in self.pairs[gp].keys():
                a, b = self.pairs[gp][lb]
                self.pairs[gp][lb] = (rx[a], rx[b])

    @Appender(CovStruct.update.__doc__)
    def update(self, params):
        endog = self.model.endog_li
        varfunc = self.model.family.variance
        cached_means = self.model.cached_means
        dep_params = defaultdict(lambda: [0.0, 0.0, 0.0])
        n_pairs = defaultdict(int)
        dim = len(params)
        for k, gp in enumerate(self.model.group_labels):
            expval, _ = cached_means[k]
            stdev = np.sqrt(varfunc(expval))
            resid = (endog[k] - expval) / stdev
            for lb in self.pairs[gp].keys():
                if not self.return_cov and lb in self._var_classes:
                    continue
                jj = self.pairs[gp][lb]
                dep_params[lb][0] += np.sum(resid[jj[0]] * resid[jj[1]])
                if not self.return_cov:
                    dep_params[lb][1] += np.sum(resid[jj[0]] ** 2)
                    dep_params[lb][2] += np.sum(resid[jj[1]] ** 2)
                n_pairs[lb] += len(jj[0])
        if self.return_cov:
            for lb in dep_params.keys():
                dep_params[lb] = dep_params[lb][0] / (n_pairs[lb] - dim)
        else:
            for lb in dep_params.keys():
                den = np.sqrt(dep_params[lb][1] * dep_params[lb][2])
                dep_params[lb] = dep_params[lb][0] / den
            for lb in self._var_classes:
                dep_params[lb] = 1.0
        self.dep_params = dep_params
        self.n_pairs = n_pairs

    @Appender(CovStruct.covariance_matrix.__doc__)
    def covariance_matrix(self, expval, index):
        dim = len(expval)
        cmat = np.zeros((dim, dim))
        g_lb = self.model.group_labels[index]
        for lb in self.pairs[g_lb].keys():
            j1, j2 = self.pairs[g_lb][lb]
            cmat[j1, j2] = self.dep_params[lb]
        cmat = cmat + cmat.T
        np.fill_diagonal(cmat, cmat.diagonal() / 2)
        return (cmat, not self.return_cov)