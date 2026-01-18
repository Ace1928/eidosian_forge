from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .base import (
from .model_selection import cross_val_predict
from .utils import Bunch, _print_elapsed_time, check_random_state
from .utils._param_validation import HasMethods, StrOptions
from .utils.metadata_routing import (
from .utils.metaestimators import available_if
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, check_is_fitted, has_fit_parameter
class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    """Multi target regression.

    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.

    .. versionadded:: 0.18

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.

        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.

        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.

        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.

    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    RegressorChain : A multi-label model that arranges regressions into a
        chain.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_linnerud
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> X, y = load_linnerud(return_X_y=True)
    >>> regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    >>> regr.predict(X[[0]])
    array([[176..., 35..., 57...]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    @_available_if_estimator_has('partial_fit')
    def partial_fit(self, X, y, sample_weight=None, **partial_fit_params):
        """Incrementally fit the model to data, for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **partial_fit_params : dict of str -> object
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().partial_fit(X, y, sample_weight=sample_weight, **partial_fit_params)