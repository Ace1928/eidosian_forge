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
def _check_set_output_transform_dataframe(name, transformer_orig, *, dataframe_lib, is_supported_dataframe, create_dataframe, assert_frame_equal, context):
    """Check that a transformer can output a DataFrame when requested.

    The DataFrame implementation is specified through the parameters of this function.

    Parameters
    ----------
    name : str
        The name of the transformer.
    transformer_orig : estimator
        The original transformer instance.
    dataframe_lib : str
        The name of the library implementing the DataFrame.
    is_supported_dataframe : callable
        A callable that takes a DataFrame instance as input and returns whether or
        not it is supported by the dataframe library.
        E.g. `lambda X: isintance(X, pd.DataFrame)`.
    create_dataframe : callable
        A callable taking as parameters `data`, `columns`, and `index` and returns
        a callable. Be aware that `index` can be ignored. For example, polars dataframes
        will ignore the index.
    assert_frame_equal : callable
        A callable taking 2 dataframes to compare if they are equal.
    context : {"local", "global"}
        Whether to use a local context by setting `set_output(...)` on the transformer
        or a global context by using the `with config_context(...)`
    """
    tags = transformer_orig._get_tags()
    if '2darray' not in tags['X_types'] or tags['no_validation']:
        return
    rng = np.random.RandomState(0)
    transformer = clone(transformer_orig)
    X = rng.uniform(size=(20, 5))
    X = _enforce_estimator_tags_X(transformer_orig, X)
    y = rng.randint(0, 2, size=20)
    y = _enforce_estimator_tags_y(transformer_orig, y)
    set_random_state(transformer)
    feature_names_in = [f'col{i}' for i in range(X.shape[1])]
    index = [f'index{i}' for i in range(X.shape[0])]
    df = create_dataframe(X, columns=feature_names_in, index=index)
    transformer_default = clone(transformer).set_output(transform='default')
    outputs_default = _output_from_fit_transform(transformer_default, name, X, df, y)
    if context == 'local':
        transformer_df = clone(transformer).set_output(transform=dataframe_lib)
        context_to_use = nullcontext()
    else:
        transformer_df = clone(transformer)
        context_to_use = config_context(transform_output=dataframe_lib)
    try:
        with context_to_use:
            outputs_df = _output_from_fit_transform(transformer_df, name, X, df, y)
    except ValueError as e:
        capitalized_lib = dataframe_lib.capitalize()
        error_message = str(e)
        assert f'{capitalized_lib} output does not support sparse data.' in error_message or 'The transformer outputs a scipy sparse matrix.' in error_message, e
        return
    for case in outputs_default:
        _check_generated_dataframe(name, case, index, outputs_default[case], outputs_df[case], is_supported_dataframe, create_dataframe, assert_frame_equal)