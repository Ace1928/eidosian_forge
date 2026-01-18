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
def _construct_instance(Estimator):
    """Construct Estimator instance if possible."""
    required_parameters = getattr(Estimator, '_required_parameters', [])
    if len(required_parameters):
        if required_parameters in (['estimator'], ['base_estimator']):
            if issubclass(Estimator, RANSACRegressor):
                estimator = Estimator(LinearRegression())
            elif issubclass(Estimator, RegressorMixin):
                estimator = Estimator(Ridge())
            elif issubclass(Estimator, SelectFromModel):
                estimator = Estimator(SGDRegressor(random_state=0))
            else:
                estimator = Estimator(LogisticRegression(C=1))
        elif required_parameters in (['estimators'],):
            if issubclass(Estimator, RegressorMixin):
                estimator = Estimator(estimators=[('est1', DecisionTreeRegressor(max_depth=3, random_state=0)), ('est2', DecisionTreeRegressor(max_depth=3, random_state=1))])
            else:
                estimator = Estimator(estimators=[('est1', DecisionTreeClassifier(max_depth=3, random_state=0)), ('est2', DecisionTreeClassifier(max_depth=3, random_state=1))])
        else:
            msg = f"Can't instantiate estimator {Estimator.__name__} parameters {required_parameters}"
            warnings.warn(msg, SkipTestWarning)
            raise SkipTest(msg)
    else:
        estimator = Estimator()
    return estimator