from unittest.mock import Mock
import numpy as np
import pytest
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding, _spectral_embedding, spectral_embedding
from sklearn.manifold._spectral_embedding import (
from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import (
from sklearn.utils.fixes import laplacian as csgraph_laplacian
def _assert_equal_with_sign_flipping(A, B, tol=0.0):
    """Check array A and B are equal with possible sign flipping on
    each columns"""
    tol_squared = tol ** 2
    for A_col, B_col in zip(A.T, B.T):
        assert np.max((A_col - B_col) ** 2) <= tol_squared or np.max((A_col + B_col) ** 2) <= tol_squared