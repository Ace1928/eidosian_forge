import warnings
from inspect import signature
from math import log
from numbers import Integral, Real
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import Bunch
from ._loss import HalfBinomialLoss
from .base import (
from .isotonic import IsotonicRegression
from .model_selection import check_cv, cross_val_predict
from .preprocessing import LabelEncoder, label_binarize
from .svm import LinearSVC
from .utils import (
from .utils._param_validation import (
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
from .utils._response import _get_response_values, _process_predict_proba
from .utils.metadata_routing import (
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import (
class _SigmoidCalibration(RegressorMixin, BaseEstimator):
    """Sigmoid regression model.

    Attributes
    ----------
    a_ : float
        The slope.

    b_ : float
        The intercept.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        X = column_or_1d(X)
        y = column_or_1d(y)
        X, y = indexable(X, y)
        self.a_, self.b_ = _sigmoid_calibration(X, y, sample_weight)
        return self

    def predict(self, T):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to predict from.

        Returns
        -------
        T_ : ndarray of shape (n_samples,)
            The predicted data.
        """
        T = column_or_1d(T)
        return expit(-(self.a_ * T + self.b_))