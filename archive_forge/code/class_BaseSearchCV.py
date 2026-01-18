import numbers
import operator
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial, reduce
from itertools import product
import numpy as np
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
from ..exceptions import NotFittedError
from ..metrics import check_scoring
from ..metrics._scorer import (
from ..utils import Bunch, check_random_state
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.metaestimators import available_if
from ..utils.parallel import Parallel, delayed
from ..utils.random import sample_without_replacement
from ..utils.validation import _check_method_params, check_is_fitted, indexable
from ._split import check_cv
from ._validation import (
class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for hyper parameter search with cross-validation."""
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit'])], 'scoring': [StrOptions(set(get_scorer_names())), callable, list, tuple, dict, None], 'n_jobs': [numbers.Integral, None], 'refit': ['boolean', str, callable], 'cv': ['cv_object'], 'verbose': ['verbose'], 'pre_dispatch': [numbers.Integral, str], 'error_score': [StrOptions({'raise'}), numbers.Real], 'return_train_score': ['boolean']}

    @abstractmethod
    def __init__(self, estimator, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=True):
        self.scoring = scoring
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def _more_tags(self):
        return {'pairwise': _safe_tags(self.estimator, 'pairwise'), '_xfail_checks': {'check_supervised_y_2d': 'DataConversionWarning not caught'}}

    def score(self, X, y=None, **params):
        """Return the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output)             or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict
            Parameters to be passed to the underlying scorer(s).

            ..versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        score : float
            The score defined by ``scoring`` if provided, and the
            ``best_estimator_.score`` method otherwise.
        """
        _check_refit(self, 'score')
        check_is_fitted(self)
        _raise_for_params(params, self, 'score')
        if _routing_enabled():
            score_params = process_routing(self, 'score', **params).scorer['score']
        else:
            score_params = dict()
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, and the estimator doesn't provide one %s" % self.best_estimator_)
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y, **score_params)
        score = self.scorer_(self.best_estimator_, X, y, **score_params)
        if self.multimetric_:
            score = score[self.refit]
        return score

    @available_if(_estimator_has('score_samples'))
    def score_samples(self, X):
        """Call score_samples on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``score_samples``.

        .. versionadded:: 0.24

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements
            of the underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            The ``best_estimator_.score_samples`` method.
        """
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has('predict'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has('predict_log_proba'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_score : ndarray of shape (n_samples,) or (n_samples, n_classes)                 or (n_samples, n_classes * (n_classes-1) / 2)
            Result of the decision function for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has('transform'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        Xt : {ndarray, sparse matrix} of shape (n_samples, n_features)
            `X` transformed in the new space based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has('inverse_transform'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Result of the `inverse_transform` function for `Xt` based on the
            estimator with the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def n_features_in_(self):
        """Number of features seen during :term:`fit`.

        Only available when `refit=True`.
        """
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError('{} object has no n_features_in_ attribute.'.format(self.__class__.__name__)) from nfe
        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        """Class labels.

        Only available when `refit=True` and the estimator is a classifier.
        """
        _estimator_has('classes_')(self)
        return self.best_estimator_.classes_

    def _run_search(self, evaluate_candidates):
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        This method, implemented in sub-classes, makes it possible to
        customize the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.
        For example, Successive Halving is implemented by calling
        `evaluate_candidates` multiples times (once per iteration of the SH
        process), each time passing a different set of candidates with `X`
        and `y` of increasing sizes.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  SucessiveHaling estimators). By default, the original `cv`
                  parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`

            It returns a dict of all results so far, formatted like
            ``cv_results_``.

            Important note (relevant whether the default cv is used or not):
            in randomized splitters, and unless the random_state parameter of
            cv was set to an int, calling cv.split() multiple times will
            yield different splits. Since cv.split() is called in
            evaluate_candidates, this means that candidates will be evaluated
            on different splits each time evaluate_candidates is called. This
            might be a methodological issue depending on the search strategy
            that you're implementing. To prevent randomized splitters from
            being used, you may use _split._yields_constant_splits()

        Examples
        --------

        ::

            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        raise NotImplementedError('_run_search not implemented.')

    def _check_refit_for_multimetric(self, scores):
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = f'For multi-metric scoring, the parameter refit must be set to a scorer key or a callable to refit an estimator with the best parameter setting on the whole data and make the best_* attributes available for that metric. If this is not needed, refit should be set to False explicitly. {self.refit!r} was passed.'
        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores
        if self.refit is not False and (not valid_refit_dict) and (not callable(self.refit)):
            raise ValueError(multimetric_refit_msg)

    @staticmethod
    def _select_best_index(refit, refit_metric, results):
        """Select index of the best combination of hyperparemeters."""
        if callable(refit):
            best_index = refit(results)
            if not isinstance(best_index, numbers.Integral):
                raise TypeError('best_index_ returned is not an integer')
            if best_index < 0 or best_index >= len(results['params']):
                raise IndexError('best_index_ index out of range')
        else:
            best_index = results[f'rank_test_{refit_metric}'].argmin()
        return best_index

    def _get_scorers(self, convert_multimetric):
        """Get the scorer(s) to be used.

        This is used in ``fit`` and ``get_metadata_routing``.

        Parameters
        ----------
        convert_multimetric : bool
            Whether to convert a dict of scorers to a _MultimetricScorer. This
            is used in ``get_metadata_routing`` to include the routing info for
            multiple scorers.

        Returns
        -------
        scorers, refit_metric
        """
        refit_metric = 'score'
        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
            if convert_multimetric and isinstance(scorers, dict):
                scorers = _MultimetricScorer(scorers=scorers, raise_exc=self.error_score == 'raise')
        return (scorers, refit_metric)

    def _get_routed_params_for_fit(self, params):
        """Get the parameters to be used for routing.

        This is a method instead of a snippet in ``fit`` since it's used twice,
        here in ``fit``, and in ``HalvingRandomSearchCV.fit``.
        """
        if _routing_enabled():
            routed_params = process_routing(self, 'fit', **params)
        else:
            params = params.copy()
            groups = params.pop('groups', None)
            routed_params = Bunch(estimator=Bunch(fit=params), splitter=Bunch(split={'groups': groups}), scorer=Bunch(score={}))
        return routed_params

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output)             or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator, the scorer,
            and the CV splitter.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        scorers, refit_metric = self._get_scorers(convert_multimetric=False)
        X, y = indexable(X, y)
        params = _check_method_params(X, params=params)
        routed_params = self._get_routed_params_for_fit(params)
        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)
        base_estimator = clone(self.estimator)
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        fit_and_score_kwargs = dict(scorer=scorers, fit_params=routed_params.estimator.fit, score_params=routed_params.scorer.score, return_train_score=self.return_train_score, return_n_test_samples=True, return_times=True, return_parameters=False, error_score=self.error_score, verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)
                if self.verbose > 0:
                    print('Fitting {0} folds for each of {1} candidates, totalling {2} fits'.format(n_splits, n_candidates, n_candidates * n_splits))
                out = parallel((delayed(_fit_and_score)(clone(base_estimator), X, y, train=train, test=test, parameters=parameters, split_progress=(split_idx, n_splits), candidate_progress=(cand_idx, n_candidates), **fit_and_score_kwargs) for (cand_idx, parameters), (split_idx, (train, test)) in product(enumerate(candidate_params), enumerate(cv.split(X, y, **routed_params.splitter.split)))))
                if len(out) < 1:
                    raise ValueError('No fits were performed. Was the CV iterator empty? Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned inconsistent results. Expected {} splits, got {}'.format(n_splits, len(out) // n_candidates))
                _warn_or_raise_about_fit_failures(out, self.error_score)
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)
                nonlocal results
                results = self._format_results(all_candidate_params, n_splits, all_out, all_more_results)
                return results
            self._run_search(evaluate_candidates)
            first_test_score = all_out[0]['test_scores']
            self.multimetric_ = isinstance(first_test_score, dict)
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
            if not callable(self.refit):
                self.best_score_ = results[f'mean_test_{refit_metric}'][self.best_index_]
            self.best_params_ = results['params'][self.best_index_]
        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(**clone(self.best_params_, safe=False))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time
            if hasattr(self.best_estimator_, 'feature_names_in_'):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_
        self.scorer_ = scorers
        self.cv_results_ = results
        self.n_splits_ = n_splits
        return self

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)
        results = dict(more_results or {})
        for key, val in results.items():
            results[key] = np.asarray(val)

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    results['split%d_%s' % (split_idx, key_name)] = array[:, split_idx]
            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            if key_name.startswith(('train_', 'test_')) and np.any(~np.isfinite(array_means)):
                warnings.warn(f'One or more of the {key_name.split('_')[0]} scores are non-finite: {array_means}', category=UserWarning)
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds
            if rank:
                if np.isnan(array_means).all():
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means, method='min').astype(np.int32, copy=False)
                results['rank_%s' % key_name] = rank_result
        _store('fit_time', out['fit_time'])
        _store('score_time', out['score_time'])
        param_results = defaultdict(partial(MaskedArray, np.empty(n_candidates), mask=True, dtype=object))
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                param_results['param_%s' % name][cand_idx] = value
        results.update(param_results)
        results['params'] = candidate_params
        test_scores_dict = _normalize_score_results(out['test_scores'])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out['train_scores'])
        for scorer_name in test_scores_dict:
            _store('test_%s' % scorer_name, test_scores_dict[scorer_name], splits=True, rank=True, weights=None)
            if self.return_train_score:
                _store('train_%s' % scorer_name, train_scores_dict[scorer_name], splits=True)
        return results

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__)
        router.add(estimator=self.estimator, method_mapping=MethodMapping().add(caller='fit', callee='fit'))
        scorer, _ = self._get_scorers(convert_multimetric=True)
        router.add(scorer=scorer, method_mapping=MethodMapping().add(caller='score', callee='score').add(caller='fit', callee='score'))
        router.add(splitter=self.cv, method_mapping=MethodMapping().add(caller='fit', callee='split'))
        return router