import numbers
import warnings
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import optimize
from sklearn.metrics import get_scorer_names
from .._loss.loss import HalfBinomialLoss, HalfMultinomialLoss
from ..base import _fit_context
from ..metrics import get_scorer
from ..model_selection import check_cv
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..svm._base import _fit_liblinear
from ..utils import (
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import row_norms, softmax
from ..utils.metadata_routing import (
from ..utils.multiclass import check_classification_targets
from ..utils.optimize import _check_optimize_result, _newton_cg
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from ._base import BaseEstimator, LinearClassifierMixin, SparseCoefMixin
from ._glm.glm import NewtonCholeskySolver
from ._linear_loss import LinearModelLoss
from ._sag import sag_solver
def _check_multi_class(multi_class, solver, n_classes):
    """Computes the multi class type, either "multinomial" or "ovr".

    For `n_classes` > 2 and a solver that supports it, returns "multinomial".
    For all other cases, in particular binary classification, return "ovr".
    """
    if multi_class == 'auto':
        if solver in ('liblinear', 'newton-cholesky'):
            multi_class = 'ovr'
        elif n_classes > 2:
            multi_class = 'multinomial'
        else:
            multi_class = 'ovr'
    if multi_class == 'multinomial' and solver in ('liblinear', 'newton-cholesky'):
        raise ValueError('Solver %s does not support a multinomial backend.' % solver)
    return multi_class