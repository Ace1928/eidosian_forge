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
def _bin_data(self, X, is_training_data):
    """Bin data X.

        If is_training_data, then fit the _bin_mapper attribute.
        Else, the binned data is converted to a C-contiguous array.
        """
    description = 'training' if is_training_data else 'validation'
    if self.verbose:
        print('Binning {:.3f} GB of {} data: '.format(X.nbytes / 1000000000.0, description), end='', flush=True)
    tic = time()
    if is_training_data:
        X_binned = self._bin_mapper.fit_transform(X)
    else:
        X_binned = self._bin_mapper.transform(X)
        X_binned = np.ascontiguousarray(X_binned)
    toc = time()
    if self.verbose:
        duration = toc - tic
        print('{:.3f} s'.format(duration))
    return X_binned