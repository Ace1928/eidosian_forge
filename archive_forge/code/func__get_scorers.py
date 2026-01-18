import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import (
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import (
def _get_scorers(self, convert_multimetric):
    """Get the scorer(s) to be used.

        This is used in ``fit`` and ``get_metadata_routing``.

        Parameters
        ----------
        convert_multimetric : bool
            Whether to convert a dict of scorers to a _MultimetricScorer. This
            is used in ``get_metadata_routing`` to include the routing info for
            multiple scorers.

        Returns
        -------
        scorers, refit_metric
        """
    refit_metric = 'score'
    if callable(self.scoring):
        scorers = self.scoring
    elif self.scoring is None or isinstance(self.scoring, str):
        scorers = check_scoring(self.estimator, self.scoring)
    else:
        scorers = _check_multimetric_scoring(self.estimator, self.scoring)
        self._check_refit_for_multimetric(scorers)
        refit_metric = self.refit
        if convert_multimetric and isinstance(scorers, dict):
            scorers = _MultimetricScorer(scorers=scorers, raise_exc=self.error_score == 'raise')
    return (scorers, refit_metric)