import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import SparseEfficiencyWarning, issparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import DataConversionWarning
from ..metrics import pairwise_distances
from ..metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
from ..neighbors import NearestNeighbors
from ..utils import gen_batches, get_chunk_n_rows
from ..utils._param_validation import (
from ..utils.validation import check_memory
Perform OPTICS clustering.

        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using ``max_eps`` distance specified at
        OPTICS object instantiation.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features), or                 (n_samples, n_samples) if metric='precomputed'
            A feature array, or array of distances between samples if
            metric='precomputed'. If a sparse matrix is provided, it will be
            converted into CSR format.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        