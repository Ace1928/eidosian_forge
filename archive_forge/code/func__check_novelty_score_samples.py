import warnings
from numbers import Real
import numpy as np
from ..base import OutlierMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
from ._base import KNeighborsMixin, NeighborsBase
def _check_novelty_score_samples(self):
    if not self.novelty:
        msg = 'score_samples is not available when novelty=False. The scores of the training samples are always available through the negative_outlier_factor_ attribute. Use novelty=True if you want to use LOF for novelty detection and compute score_samples for new unseen data.'
        raise AttributeError(msg)
    return True