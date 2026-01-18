import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from ..base import (
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEnsemble, _partition_estimators
def _set_oob_score_and_attributes(self, X, y, scoring_function=None):
    raise NotImplementedError('OOB score not supported by tree embedding')