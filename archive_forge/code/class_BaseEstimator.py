import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
class BaseEstimator(_HTMLDocumentationLinkMixin, _MetadataRequester):
    """Base class for all estimators in scikit-learn.

    Inheriting from this class provides default implementations of:

    - setting and getting parameters used by `GridSearchCV` and friends;
    - textual and HTML representation displayed in terminals and IDEs;
    - estimator serialization;
    - parameters validation;
    - data validation;
    - feature names validation.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.


    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> class MyEstimator(BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=2)
    >>> estimator.get_params()
    {'param': 2}
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([2, 2, 2])
    >>> estimator.set_params(param=3).fit(X, y).predict(X)
    array([3, 3, 3])
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always specify their parameters in the signature of their __init__ (no varargs). %s with constructor %s doesn't  follow this convention." % (cls, init_signature))
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params') and (not isinstance(value, type)):
                deep_items = value.get_params().items()
                out.update(((key + '__' + k, val) for k, val in deep_items))
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(f'Invalid parameter {key!r} for estimator {self}. Valid parameters are: {local_valid_params!r}.')
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

    def __sklearn_clone__(self):
        return _clone_parametrized(self)

    def __repr__(self, N_CHAR_MAX=700):
        from .utils._pprint import _EstimatorPrettyPrinter
        N_MAX_ELEMENTS_TO_SHOW = 30
        pp = _EstimatorPrettyPrinter(compact=True, indent=1, indent_at_name=True, n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)
        repr_ = pp.pformat(self)
        n_nonblank = len(''.join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2
            regex = '^(\\s*\\S){%d}' % lim
            left_lim = re.match(regex, repr_).end()
            right_lim = re.match(regex, repr_[::-1]).end()
            if '\n' in repr_[left_lim:-right_lim]:
                regex += '[^\\n]*\\n'
                right_lim = re.match(regex, repr_[::-1]).end()
            ellipsis = '...'
            if left_lim + len(ellipsis) < len(repr_) - right_lim:
                repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]
        return repr_

    def __getstate__(self):
        if getattr(self, '__slots__', None):
            raise TypeError('You cannot use `__slots__` in objects inheriting from `sklearn.base.BaseEstimator`.')
        try:
            state = super().__getstate__()
            if state is None:
                state = self.__dict__.copy()
        except AttributeError:
            state = self.__dict__.copy()
        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            return state

    def __setstate__(self, state):
        if type(self).__module__.startswith('sklearn.'):
            pickle_version = state.pop('_sklearn_version', 'pre-0.18')
            if pickle_version != __version__:
                warnings.warn(InconsistentVersionWarning(estimator_name=self.__class__.__name__, current_sklearn_version=__version__, original_sklearn_version=pickle_version))
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)

    def _more_tags(self):
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        try:
            n_features = _num_features(X)
        except TypeError as e:
            if not reset and hasattr(self, 'n_features_in_'):
                raise ValueError(f'X does not contain any features, but {self.__class__.__name__} is expecting {self.n_features_in_} features') from e
            return
        if reset:
            self.n_features_in_ = n_features
            return
        if not hasattr(self, 'n_features_in_'):
            return
        if n_features != self.n_features_in_:
            raise ValueError(f'X has {n_features} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input.')

    def _check_feature_names(self, X, *, reset):
        """Set or check the `feature_names_in_` attribute.

        .. versionadded:: 1.0

        Parameters
        ----------
        X : {ndarray, dataframe} of shape (n_samples, n_features)
            The input samples.

        reset : bool
            Whether to reset the `feature_names_in_` attribute.
            If False, the input will be checked for consistency with
            feature names of data provided when reset was last True.
            .. note::
               It is recommended to call `reset=True` in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        if reset:
            feature_names_in = _get_feature_names(X)
            if feature_names_in is not None:
                self.feature_names_in_ = feature_names_in
            elif hasattr(self, 'feature_names_in_'):
                delattr(self, 'feature_names_in_')
            return
        fitted_feature_names = getattr(self, 'feature_names_in_', None)
        X_feature_names = _get_feature_names(X)
        if fitted_feature_names is None and X_feature_names is None:
            return
        if X_feature_names is not None and fitted_feature_names is None:
            warnings.warn(f'X has feature names, but {self.__class__.__name__} was fitted without feature names')
            return
        if X_feature_names is None and fitted_feature_names is not None:
            warnings.warn(f'X does not have valid feature names, but {self.__class__.__name__} was fitted with feature names')
            return
        if len(fitted_feature_names) != len(X_feature_names) or np.any(fitted_feature_names != X_feature_names):
            message = 'The feature names should match those that were passed during fit.\n'
            fitted_feature_names_set = set(fitted_feature_names)
            X_feature_names_set = set(X_feature_names)
            unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
            missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

            def add_names(names):
                output = ''
                max_n_names = 5
                for i, name in enumerate(names):
                    if i >= max_n_names:
                        output += '- ...\n'
                        break
                    output += f'- {name}\n'
                return output
            if unexpected_names:
                message += 'Feature names unseen at fit time:\n'
                message += add_names(unexpected_names)
            if missing_names:
                message += 'Feature names seen at fit time, yet now missing:\n'
                message += add_names(missing_names)
            if not missing_names and (not unexpected_names):
                message += 'Feature names must be in the same order as they were in fit.\n'
            raise ValueError(message)

    def _validate_data(self, X='no_validation', y='no_validation', reset=True, validate_separately=False, cast_to_ndarray=True, **check_params):
        """Validate input data and set or check the `n_features_in_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape                 (n_samples, n_features), default='no validation'
            The input samples.
            If `'no_validation'`, no validation is performed on `X`. This is
            useful for meta-estimator which can delegate input validation to
            their underlying estimator(s). In that case `y` must be passed and
            the only accepted `check_params` are `multi_output` and
            `y_numeric`.

        y : array-like of shape (n_samples,), default='no_validation'
            The targets.

            - If `None`, `check_array` is called on `X`. If the estimator's
              requires_y tag is True, then an error will be raised.
            - If `'no_validation'`, `check_array` is called on `X` and the
              estimator's requires_y tag is ignored. This is a default
              placeholder and is never meant to be explicitly set. In that case
              `X` must be passed.
            - Otherwise, only `y` with `_check_y` or both `X` and `y` are
              checked with either `check_array` or `check_X_y` depending on
              `validate_separately`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.

        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.

            `estimator=self` is automatically added to these dicts to generate
            more informative error message in case of invalid input data.

        cast_to_ndarray : bool, default=True
            Cast `X` and `y` to ndarray with checks in `check_params`. If
            `False`, `X` and `y` are unchanged and only `feature_names_in_` and
            `n_features_in_` are checked.

        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

            `estimator=self` is automatically added to these params to generate
            more informative error message in case of invalid input data.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        self._check_feature_names(X, reset=reset)
        if y is None and self._get_tags()['requires_y']:
            raise ValueError(f'This {self.__class__.__name__} estimator requires y to be passed, but the target y is None.')
        no_val_X = isinstance(X, str) and X == 'no_validation'
        no_val_y = y is None or (isinstance(y, str) and y == 'no_validation')
        if no_val_X and no_val_y:
            raise ValueError('Validation should be done on X, y or both.')
        default_check_params = {'estimator': self}
        check_params = {**default_check_params, **check_params}
        if not cast_to_ndarray:
            if not no_val_X and no_val_y:
                out = X
            elif no_val_X and (not no_val_y):
                out = y
            else:
                out = (X, y)
        elif not no_val_X and no_val_y:
            out = check_array(X, input_name='X', **check_params)
        elif no_val_X and (not no_val_y):
            out = _check_y(y, **check_params)
        else:
            if validate_separately:
                check_X_params, check_y_params = validate_separately
                if 'estimator' not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, input_name='X', **check_X_params)
                if 'estimator' not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name='y', **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = (X, y)
        if not no_val_X and check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)
        return out

    def _validate_params(self):
        """Validate types and values of constructor parameters

        The expected type and values must be defined in the `_parameter_constraints`
        class attribute, which is a dictionary `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for a description of the
        accepted constraints.
        """
        validate_parameter_constraints(self._parameter_constraints, self.get_params(deep=False), caller_name=self.__class__.__name__)

    @property
    def _repr_html_(self):
        """HTML representation of estimator.

        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favorted in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimbundle_`.
        """
        if get_config()['display'] != 'diagram':
            raise AttributeError("_repr_html_ is only defined when the 'display' configuration option is set to 'diagram'")
        return self._repr_html_inner

    def _repr_html_inner(self):
        """This function is returned by the @property `_repr_html_` to make
        `hasattr(estimator, "_repr_html_") return `True` or `False` depending
        on `get_config()["display"]`.
        """
        return estimator_html_repr(self)

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display estimator"""
        output = {'text/plain': repr(self)}
        if get_config()['display'] == 'diagram':
            output['text/html'] = estimator_html_repr(self)
        return output