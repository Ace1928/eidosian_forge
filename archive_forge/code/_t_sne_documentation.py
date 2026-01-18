from numbers import Integral, Real
from time import time
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, squareform
from ..base import (
from ..decomposition import PCA
from ..metrics.pairwise import _VALID_METRICS, pairwise_distances
from ..neighbors import NearestNeighbors
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import _num_samples, check_non_negative
from . import _barnes_hut_tsne, _utils  # type: ignore
Number of transformed output features.