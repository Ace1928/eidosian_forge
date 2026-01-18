from itertools import product
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import config_context, datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.utils._array_api import yield_namespace_device_dtype_combinations
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import (
def _generate_sparse(data, sparse_containers=tuple(COO_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS), dtypes=(bool, int, np.int8, np.uint8, float, np.float32)):
    return [sparse_container(data, dtype=dtype) for sparse_container in sparse_containers for dtype in dtypes]