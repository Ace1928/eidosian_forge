import pickle
import re
import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial, wraps
from inspect import signature
from numbers import Integral, Real
import joblib
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from .. import config_context
from ..base import (
from ..datasets import (
from ..exceptions import DataConversionWarning, NotFittedError, SkipTestWarning
from ..feature_selection import SelectFromModel, SelectKBest
from ..linear_model import (
from ..metrics import accuracy_score, adjusted_rand_score, f1_score
from ..metrics.pairwise import linear_kernel, pairwise_distances, rbf_kernel
from ..model_selection import ShuffleSplit, train_test_split
from ..model_selection._validation import _safe_split
from ..pipeline import make_pipeline
from ..preprocessing import StandardScaler, scale
from ..random_projection import BaseRandomProjection
from ..tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..utils._array_api import (
from ..utils._array_api import (
from ..utils._param_validation import (
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import check_is_fitted
from . import IS_PYPY, is_scalar_nan, shuffle
from ._param_validation import Interval
from ._tags import (
from ._testing import (
from .validation import _num_samples, has_fit_parameter
def _output_from_fit_transform(transformer, name, X, df, y):
    """Generate output to test `set_output` for different configuration:

    - calling either `fit.transform` or `fit_transform`;
    - passing either a dataframe or a numpy array to fit;
    - passing either a dataframe or a numpy array to transform.
    """
    outputs = {}
    cases = [('fit.transform/df/df', df, df), ('fit.transform/df/array', df, X), ('fit.transform/array/df', X, df), ('fit.transform/array/array', X, X)]
    if all((hasattr(transformer, meth) for meth in ['fit', 'transform'])):
        for case, data_fit, data_transform in cases:
            transformer.fit(data_fit, y)
            if name in CROSS_DECOMPOSITION:
                X_trans, _ = transformer.transform(data_transform, y)
            else:
                X_trans = transformer.transform(data_transform)
            outputs[case] = (X_trans, transformer.get_feature_names_out())
    cases = [('fit_transform/df', df), ('fit_transform/array', X)]
    if hasattr(transformer, 'fit_transform'):
        for case, data in cases:
            if name in CROSS_DECOMPOSITION:
                X_trans, _ = transformer.fit_transform(data, y)
            else:
                X_trans = transformer.fit_transform(data, y)
            outputs[case] = (X_trans, transformer.get_feature_names_out())
    return outputs