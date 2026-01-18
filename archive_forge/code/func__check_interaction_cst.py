import itertools
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, suppress
from functools import partial
from numbers import Integral, Real
from time import time
import numpy as np
from ..._loss.loss import (
from ...base import (
from ...compose import ColumnTransformer
from ...metrics import check_scoring
from ...metrics._scorer import _SCORERS
from ...model_selection import train_test_split
from ...preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from ...utils import check_random_state, compute_sample_weight, is_scalar_nan, resample
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ...utils.multiclass import check_classification_targets
from ...utils.validation import (
from ._gradient_boosting import _update_raw_predictions
from .binning import _BinMapper
from .common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from .grower import TreeGrower
def _check_interaction_cst(self, n_features):
    """Check and validation for interaction constraints."""
    if self.interaction_cst is None:
        return None
    if self.interaction_cst == 'no_interactions':
        interaction_cst = [[i] for i in range(n_features)]
    elif self.interaction_cst == 'pairwise':
        interaction_cst = itertools.combinations(range(n_features), 2)
    else:
        interaction_cst = self.interaction_cst
    try:
        constraints = [set(group) for group in interaction_cst]
    except TypeError:
        raise ValueError(f'Interaction constraints must be a sequence of tuples or lists, got: {self.interaction_cst!r}.')
    for group in constraints:
        for x in group:
            if not (isinstance(x, Integral) and 0 <= x < n_features):
                raise ValueError(f'Interaction constraints must consist of integer indices in [0, n_features - 1] = [0, {n_features - 1}], specifying the position of features, got invalid indices: {group!r}')
    rest = set(range(n_features)) - set().union(*constraints)
    if len(rest) > 0:
        constraints.append(rest)
    return constraints