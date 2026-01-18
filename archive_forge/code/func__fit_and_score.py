import numbers
import time
import warnings
from collections import Counter
from contextlib import suppress
from functools import partial
from numbers import Real
from traceback import format_exc
import numpy as np
import scipy.sparse as sp
from joblib import logger
from ..base import clone, is_classifier
from ..exceptions import FitFailedWarning, UnsetMetadataPassedError
from ..metrics import check_scoring, get_scorer_names
from ..metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from ..preprocessing import LabelEncoder
from ..utils import Bunch, _safe_indexing, check_random_state, indexable
from ..utils._param_validation import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _safe_split
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_method_params, _num_samples
from ._split import check_cv
def _fit_and_score(estimator, X, y, *, scorer, train, test, verbose, parameters, fit_params, score_params, return_train_score=False, return_parameters=False, return_n_test_samples=False, return_times=False, return_estimator=False, split_progress=None, candidate_progress=None, error_score=np.nan):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    score_params : dict or None
        Parameters that will be passed to the scorer.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    if not isinstance(error_score, numbers.Number) and error_score != 'raise':
        raise ValueError("error_score must be the string 'raise' or a numeric value. (Hint: if using 'raise', please make sure that it has been spelled correctly.)")
    progress_msg = ''
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f' {split_progress[0] + 1}/{split_progress[1]}'
        if candidate_progress and verbose > 9:
            progress_msg += f'; {candidate_progress[0] + 1}/{candidate_progress[1]}'
    if verbose > 1:
        if parameters is None:
            params_msg = ''
        else:
            sorted_keys = sorted(parameters)
            params_msg = ', '.join((f'{k}={parameters[k]}' for k in sorted_keys))
    if verbose > 9:
        start_msg = f'[CV{progress_msg}] START {params_msg}'
        print(f'{start_msg}{(80 - len(start_msg)) * '.'}')
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    score_params = score_params if score_params is not None else {}
    score_params_train = _check_method_params(X, params=score_params, indices=train)
    score_params_test = _check_method_params(X, params=score_params, indices=test)
    if parameters is not None:
        estimator = estimator.set_params(**clone(parameters, safe=False))
    start_time = time.time()
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)
    except Exception:
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result['fit_error'] = format_exc()
    else:
        result['fit_error'] = None
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer, score_params_test, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer, score_params_train, error_score)
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f'[CV{progress_msg}] END '
        result_msg = params_msg + (';' if params_msg else '')
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f' {scorer_name}: ('
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f'train={scorer_scores:.3f}, '
                    result_msg += f'test={test_scores[scorer_name]:.3f})'
            else:
                result_msg += ', score='
                if return_train_score:
                    result_msg += f'(train={train_scores:.3f}, test={test_scores:.3f})'
                else:
                    result_msg += f'{test_scores:.3f}'
        result_msg += f' total time={logger.short_format_time(total_time)}'
        end_msg += '.' * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)
    result['test_scores'] = test_scores
    if return_train_score:
        result['train_scores'] = train_scores
    if return_n_test_samples:
        result['n_test_samples'] = _num_samples(X_test)
    if return_times:
        result['fit_time'] = fit_time
        result['score_time'] = score_time
    if return_parameters:
        result['parameters'] = parameters
    if return_estimator:
        result['estimator'] = estimator
    return result