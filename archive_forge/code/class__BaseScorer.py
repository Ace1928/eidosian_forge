import copy
import warnings
from collections import Counter
from functools import partial
from inspect import signature
from traceback import format_exc
from ..base import is_regressor
from ..utils import Bunch
from ..utils._param_validation import HasMethods, Hidden, StrOptions, validate_params
from ..utils._response import _get_response_values
from ..utils.metadata_routing import (
from ..utils.validation import _check_response_method
from . import (
from .cluster import (
class _BaseScorer(_MetadataRequester):

    def __init__(self, score_func, sign, kwargs, response_method='predict'):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method

    def _get_pos_label(self):
        if 'pos_label' in self._kwargs:
            return self._kwargs['pos_label']
        score_func_params = signature(self._score_func).parameters
        if 'pos_label' in score_func_params:
            return score_func_params['pos_label'].default
        return None

    def __repr__(self):
        sign_string = '' if self._sign > 0 else ', greater_is_better=False'
        response_method_string = f', response_method={self._response_method!r}'
        kwargs_string = ''.join([f', {k}={v}' for k, v in self._kwargs.items()])
        return f'make_scorer({self._score_func.__name__}{sign_string}{response_method_string}{kwargs_string})'

    def __call__(self, estimator, X, y_true, sample_weight=None, **kwargs):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        **kwargs : dict
            Other parameters passed to the scorer. Refer to
            :func:`set_score_request` for more details.

            Only available if `enable_metadata_routing=True`. See the
            :ref:`User Guide <metadata_routing>`.

            .. versionadded:: 1.3

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        _raise_for_params(kwargs, self, None)
        _kwargs = copy.deepcopy(kwargs)
        if sample_weight is not None:
            _kwargs['sample_weight'] = sample_weight
        return self._score(partial(_cached_call, None), estimator, X, y_true, **_kwargs)

    def _warn_overlap(self, message, kwargs):
        """Warn if there is any overlap between ``self._kwargs`` and ``kwargs``.

        This method is intended to be used to check for overlap between
        ``self._kwargs`` and ``kwargs`` passed as metadata.
        """
        _kwargs = set() if self._kwargs is None else set(self._kwargs.keys())
        overlap = _kwargs.intersection(kwargs.keys())
        if overlap:
            warnings.warn(f'{message} Overlapping parameters are: {overlap}', UserWarning)

    def set_score_request(self, **kwargs):
        """Set requested parameters by the scorer.

        Please see :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.3

        Parameters
        ----------
        kwargs : dict
            Arguments should be of the form ``param_name=alias``, and `alias`
            can be one of ``{True, False, None, str}``.
        """
        if not _routing_enabled():
            raise RuntimeError('This method is only available when metadata routing is enabled. You can enable it using sklearn.set_config(enable_metadata_routing=True).')
        self._warn_overlap(message='You are setting metadata request for parameters which are already set as kwargs for this metric. These set values will be overridden by passed metadata if provided. Please pass them either as metadata or kwargs to `make_scorer`.', kwargs=kwargs)
        self._metadata_request = MetadataRequest(owner=self.__class__.__name__)
        for param, alias in kwargs.items():
            self._metadata_request.score.add_request(param=param, alias=alias)
        return self