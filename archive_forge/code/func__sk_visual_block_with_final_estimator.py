from abc import ABCMeta, abstractmethod
from copy import deepcopy
from numbers import Integral
import numpy as np
import scipy.sparse as sparse
from ..base import (
from ..exceptions import NotFittedError
from ..linear_model import LogisticRegression, RidgeCV
from ..model_selection import check_cv, cross_val_predict
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._param_validation import HasMethods, StrOptions
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
def _sk_visual_block_with_final_estimator(self, final_estimator):
    names, estimators = zip(*self.estimators)
    parallel = _VisualBlock('parallel', estimators, names=names, dash_wrapped=False)
    final_block = _VisualBlock('parallel', [final_estimator], names=['final_estimator'], dash_wrapped=False)
    return _VisualBlock('serial', (parallel, final_block), dash_wrapped=False)