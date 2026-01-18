from collections import defaultdict
from itertools import islice
import numpy as np
from scipy import sparse
from .base import TransformerMixin, _fit_context, clone
from .exceptions import NotFittedError
from .preprocessing import FunctionTransformer
from .utils import Bunch, _print_elapsed_time
from .utils._estimator_html_repr import _VisualBlock
from .utils._metadata_requests import METHODS
from .utils._param_validation import HasMethods, Hidden
from .utils._set_output import (
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _BaseComposition, available_if
from .utils.parallel import Parallel, delayed
from .utils.validation import check_is_fitted, check_memory
class FeatureUnion(_RoutingNotSupportedMixin, TransformerMixin, _BaseComposition):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer, removed by
    setting to 'drop' or disabled by setting to 'passthrough' (features are
    passed without transformation).

    Read more in the :ref:`User Guide <feature_union>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The transformer can
        be 'drop' for it to be ignored or can be 'passthrough' for features to
        be passed unchanged.

        .. versionadded:: 1.1
           Added the option `"passthrough"`.

        .. versionchanged:: 0.22
           Deprecated `None` as a transformer in favor of 'drop'.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Attributes
    ----------
    named_transformers : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any transformer parameter by user
        given name. Keys are transformer names and values are
        transformer parameters.

        .. versionadded:: 1.2

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first transformer in `transformer_list` exposes such an
        attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when
        `X` has feature names that are all strings.

        .. versionadded:: 1.3

    See Also
    --------
    make_union : Convenience function for simplified feature union
        construction.

    Examples
    --------
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> union = FeatureUnion([("pca", PCA(n_components=1)),
    ...                       ("svd", TruncatedSVD(n_components=2))])
    >>> X = [[0., 1., 3], [2., 2., 5]]
    >>> union.fit_transform(X)
    array([[ 1.5       ,  3.0...,  0.8...],
           [-1.5       ,  5.7..., -0.4...]])
    >>> # An estimator's parameter can be set using '__' syntax
    >>> union.set_params(svd__n_components=1).fit_transform(X)
    array([[ 1.5       ,  3.0...],
           [-1.5       ,  5.7...]])

    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_compose_plot_feature_union.py`.
    """
    _required_parameters = ['transformer_list']

    def __init__(self, transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

    def set_output(self, *, transform=None):
        """Set the output container when `"transform"` and `"fit_transform"` are called.

        `set_output` will set the output of all estimators in `transformer_list`.

        Parameters
        ----------
        transform : {"default", "pandas"}, default=None
            Configure output of `transform` and `fit_transform`.

            - `"default"`: Default output format of a transformer
            - `"pandas"`: DataFrame output
            - `None`: Transform configuration is unchanged

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        super().set_output(transform=transform)
        for _, step, _ in self._iter():
            _safe_set_output(step, transform=transform)
        return self

    @property
    def named_transformers(self):
        return Bunch(**dict(self.transformer_list))

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `transformer_list` of the
        `FeatureUnion`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('transformer_list', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `transformer_list`.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `transform_list`. Parameters of the transformers may be set
            using its name and the parameter name separated by a '__'.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        self._set_params('transformer_list', **kwargs)
        return self

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)
        self._validate_names(names)
        for t in transformers:
            if t in ('drop', 'passthrough'):
                continue
            if not (hasattr(t, 'fit') or hasattr(t, 'fit_transform')) or not hasattr(t, 'transform'):
                raise TypeError("All estimators should implement fit and transform. '%s' (type %s) doesn't" % (t, type(t)))

    def _validate_transformer_weights(self):
        if not self.transformer_weights:
            return
        transformer_names = set((name for name, _ in self.transformer_list))
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(f'Attempting to weight transformer "{name}", but it is not present in transformer_list.')

    def _iter(self):
        """
        Generate (name, trans, weight) tuples excluding None and
        'drop' transformers.
        """
        get_weight = (self.transformer_weights or {}).get
        for name, trans in self.transformer_list:
            if trans == 'drop':
                continue
            if trans == 'passthrough':
                trans = FunctionTransformer(feature_names_out='one-to-one')
            yield (name, trans, get_weight(name))

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        feature_names = []
        for name, trans, _ in self._iter():
            if not hasattr(trans, 'get_feature_names_out'):
                raise AttributeError('Transformer %s (type %s) does not provide get_feature_names_out.' % (str(name), type(trans).__name__))
            feature_names.extend([f'{name}__{f}' for f in trans.get_feature_names_out(input_features)])
        return np.asarray(feature_names, dtype=object)

    def fit(self, X, y=None, **fit_params):
        """Fit all transformers using X.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data, used to fit transformers.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        self : object
            FeatureUnion class instance.
        """
        _raise_for_unsupported_routing(self, 'fit', **fit_params)
        transformers = self._parallel_func(X, y, fit_params, _fit_one)
        if not transformers:
            return self
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.

        Returns
        -------
        X_t : array-like or sparse matrix of                 shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)
        return self._hstack(Xs)

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(step %d of %d) Processing %s' % (idx, total, name)

    def _parallel_func(self, X, y, fit_params, func):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        self._validate_transformer_weights()
        transformers = list(self._iter())
        params = Bunch(fit=fit_params, fit_transform=fit_params)
        return Parallel(n_jobs=self.n_jobs)((delayed(func)(transformer, X, y, weight, message_clsname='FeatureUnion', message=self._log_message(name, idx, len(transformers)), params=params) for idx, (name, transformer, weight) in enumerate(transformers, 1)))

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix of                 shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        params = Bunch(transform={})
        Xs = Parallel(n_jobs=self.n_jobs)((delayed(_transform_one)(trans, X, None, weight, params) for name, trans, weight in self._iter()))
        if not Xs:
            return np.zeros((X.shape[0], 0))
        return self._hstack(Xs)

    def _hstack(self, Xs):
        adapter = _get_container_adapter('transform', self)
        if adapter and all((adapter.is_supported_container(X) for X in Xs)):
            return adapter.hstack(Xs)
        if any((sparse.issparse(f) for f in Xs)):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [(name, old if old == 'drop' else next(transformers)) for name, old in self.transformer_list]

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`."""
        return self.transformer_list[0][1].n_features_in_

    @property
    def feature_names_in_(self):
        """Names of features seen during :term:`fit`."""
        return self.transformer_list[0][1].feature_names_in_

    def __sklearn_is_fitted__(self):
        for _, transformer, _ in self._iter():
            check_is_fitted(transformer)
        return True

    def _sk_visual_block_(self):
        names, transformers = zip(*self.transformer_list)
        return _VisualBlock('parallel', transformers, names=names)

    def __getitem__(self, name):
        """Return transformer with name."""
        if not isinstance(name, str):
            raise KeyError('Only string keys are supported')
        return self.named_transformers[name]