import warnings
import numpy as np
from scipy.sparse import issparse
from ..base import TransformerMixin
from ..utils import metadata_routing
from ..utils.validation import check_is_fitted

        Inverse the transformation and return a vector of size `n_features`.

        Parameters
        ----------
        Xt : array-like of shape (n_samples, n_clusters) or (n_clusters,)
            The values to be assigned to each cluster of samples.

        Xred : deprecated
            Use `Xt` instead.

            .. deprecated:: 1.3

        Returns
        -------
        X : ndarray of shape (n_samples, n_features) or (n_features,)
            A vector of size `n_samples` with the values of `Xred` assigned to
            each of the cluster of samples.
        