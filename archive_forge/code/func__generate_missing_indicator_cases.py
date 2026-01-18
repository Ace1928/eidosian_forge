import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def _generate_missing_indicator_cases():
    missing_values_dtypes = [(0, np.int32), (np.nan, np.float64), (-1, np.int32)]
    arr_types = [np.array] + CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS
    return [(arr_type, missing_values, dtype) for arr_type, (missing_values, dtype) in product(arr_types, missing_values_dtypes) if not (missing_values == 0 and arr_type is not np.array)]