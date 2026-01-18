import warnings
from numbers import Real
import numpy as np
from ..base import OutlierMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase
def _check_novelty_decision_function(self):
    if not self.novelty:
        msg = 'decision_function is not available when novelty=False. Use novelty=True if you want to use LOF for novelty detection and compute decision_function for new unseen data. Note that the opposite LOF of the training samples is always available by considering the negative_outlier_factor_ attribute.'
        raise AttributeError(msg)
    return True