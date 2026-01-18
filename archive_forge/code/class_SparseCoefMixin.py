import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from ..base import (
from ..utils import check_array, check_random_state
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
from ..utils.extmath import safe_sparse_dot
from ..utils.parallel import Parallel, delayed
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted
class SparseCoefMixin:
    """Mixin for converting coef_ to and from CSR format.

    L1-regularizing estimators should inherit this.
    """

    def densify(self):
        """
        Convert coefficient matrix to dense array format.

        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.

        Returns
        -------
        self
            Fitted estimator.
        """
        msg = 'Estimator, %(name)s, must be fitted before densifying.'
        check_is_fitted(self, msg=msg)
        if sp.issparse(self.coef_):
            self.coef_ = self.coef_.toarray()
        return self

    def sparsify(self):
        """
        Convert coefficient matrix to sparse format.

        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.

        The ``intercept_`` member is not converted.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.

        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.
        """
        msg = 'Estimator, %(name)s, must be fitted before sparsifying.'
        check_is_fitted(self, msg=msg)
        self.coef_ = sp.csr_matrix(self.coef_)
        return self