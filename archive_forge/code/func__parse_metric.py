import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
def _parse_metric(metric: str, dtype=None):
    """
    Helper function for properly building a type-specialized DistanceMetric instances.

    Constructs a type-specialized DistanceMetric instance from a string
    beginning with "DM_" while allowing a pass-through for other metric-specifying
    strings. This is necessary since we wish to parameterize dtype independent of
    metric, yet DistanceMetric requires it for construction.

    """
    if metric[:3] == 'DM_':
        return DistanceMetric.get_metric(metric[3:], dtype=dtype)
    return metric