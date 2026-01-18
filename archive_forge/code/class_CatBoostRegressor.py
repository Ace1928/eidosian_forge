from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
class CatBoostRegressor(CatBoost):
    """
    Implementation of the scikit-learn API for CatBoost regression.

    Parameters
    ----------
    Like in CatBoostClassifier, except loss_function, classes_count, class_names and class_weights

    loss_function : string, [default='RMSE']
        'RMSE'
        'MAE'
        'Quantile:alpha=value'
        'LogLinQuantile:alpha=value'
        'Poisson'
        'MAPE'
        'Lq:q=value'
        'SurvivalAft:dist=value;scale=value'
    """
    _estimator_type = 'regressor'

    def __init__(self, iterations=None, learning_rate=None, depth=None, l2_leaf_reg=None, model_size_reg=None, rsm=None, loss_function='RMSE', border_count=None, feature_border_type=None, per_float_feature_quantization=None, input_borders=None, output_borders=None, fold_permutation_block=None, od_pval=None, od_wait=None, od_type=None, nan_mode=None, counter_calc_method=None, leaf_estimation_iterations=None, leaf_estimation_method=None, thread_count=None, random_seed=None, use_best_model=None, best_model_min_trees=None, verbose=None, silent=None, logging_level=None, metric_period=None, ctr_leaf_count_limit=None, store_all_simple_ctr=None, max_ctr_complexity=None, has_time=None, allow_const_label=None, target_border=None, one_hot_max_size=None, random_strength=None, random_score_type=None, name=None, ignored_features=None, train_dir=None, custom_metric=None, eval_metric=None, bagging_temperature=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, fold_len_multiplier=None, used_ram_limit=None, gpu_ram_part=None, pinned_memory_size=None, allow_writing_files=None, final_ctr_computation_mode=None, approx_on_full_history=None, boosting_type=None, simple_ctr=None, combinations_ctr=None, per_feature_ctr=None, ctr_description=None, ctr_target_border_count=None, task_type=None, device_config=None, devices=None, bootstrap_type=None, subsample=None, mvs_reg=None, sampling_frequency=None, sampling_unit=None, dev_score_calc_obj_block_size=None, dev_efb_max_buckets=None, sparse_features_conflict_fraction=None, max_depth=None, n_estimators=None, num_boost_round=None, num_trees=None, colsample_bylevel=None, random_state=None, reg_lambda=None, objective=None, eta=None, max_bin=None, gpu_cat_features_storage=None, data_partition=None, metadata=None, early_stopping_rounds=None, cat_features=None, grow_policy=None, min_data_in_leaf=None, min_child_samples=None, max_leaves=None, num_leaves=None, score_function=None, leaf_estimation_backtracking=None, ctr_history_unit=None, monotone_constraints=None, feature_weights=None, penalties_coefficient=None, first_feature_use_penalties=None, per_object_feature_penalties=None, model_shrink_rate=None, model_shrink_mode=None, langevin=None, diffusion_temperature=None, posterior_sampling=None, boost_from_average=None, text_features=None, tokenizers=None, dictionaries=None, feature_calcers=None, text_processing=None, embedding_features=None, eval_fraction=None, fixed_binary_splits=None):
        params = {}
        not_params = ['not_params', 'self', 'params', '__class__']
        for key, value in iteritems(locals().copy()):
            if key not in not_params and value is not None:
                params[key] = value
        super(CatBoostRegressor, self).__init__(params)

    def fit(self, X, y=None, cat_features=None, text_features=None, embedding_features=None, sample_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None, log_cout=None, log_cerr=None):
        """
        Fit the CatBoost model.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            If not catboost.Pool, 2 dimensional Feature matrix or string - file with dataset.

        y : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Labels of the training data.
            If not None, can be a single- or two- dimensional array with numerical values.
            Use only if X is not catboost.Pool and does not point to a file.

        cat_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Categ columns indices.
            Use only if X is not catboost.Pool.

        text_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Text columns indices.
            Use only if X is not catboost.Pool.

        embedding_features : list or numpy.ndarray, optional (default=None)
            If not None, giving the list of Embedding columns indices.
            Use only if X is not catboost.Pool.

        sample_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Instance weights, 1 dimensional array like.

        baseline : list or numpy.ndarray, optional (default=None)
            If not None, giving 2 dimensional array like data.
            Use only if X is not catboost.Pool.

        use_best_model : bool, optional (default=None)
            Flag to use best model

        eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
            Validation dataset or datasets for metrics calculation and possibly early stopping.

        metric_period : int
            Frequency of evaluating metrics.

        verbose : bool or int
            If verbose is bool, then if set to True, logging_level is set to Verbose,
            if set to False, logging_level is set to Silent.
            If verbose is int, it determines the frequency of writing metrics to output and
            logging_level is set to Verbose.

        silent : bool
            If silent is True, logging_level is set to Silent.
            If silent is False, logging_level is set to Verbose.

        logging_level : string, optional (default=None)
            Possible values:
                - 'Silent'
                - 'Verbose'
                - 'Info'
                - 'Debug'

        plot : bool, optional (default=False)
            If True, draw train and eval error in Jupyter notebook

        plot_file : file-like or str, optional (default=None)
            If not None, save train and eval error graphs to file (requires installed plotly)

        verbose_eval : bool or int
            Synonym for verbose. Only one of these parameters should be set.

        early_stopping_rounds : int
            Activates Iter overfitting detector with od_wait set to early_stopping_rounds.

        save_snapshot : bool, [default=None]
            Enable progress snapshotting for restoring progress after crashes or interruptions

        snapshot_file : string or pathlib.Path, [default=None]
            Learn progress snapshot file path, if None will use default filename

        snapshot_interval: int, [default=600]
            Interval between saving snapshots (seconds)

        init_model : CatBoost class or string or pathlib.Path, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string or pathlib.Path, load initial model from the path specified by this string.

        callbacks : list, optional (default=None)
            List of callback objects that are applied at end of each iteration.

        log_cout: output stream or callback for logging (default=None)
            If None is specified, sys.stdout is used

        log_cerr: error stream or callback for logging (default=None)
            If None is specified, sys.stderr is used

        Returns
        -------
        model : CatBoost
        """
        params = deepcopy(self._init_params)
        _process_synonyms(params)
        if 'loss_function' in params:
            CatBoostRegressor._check_is_compatible_loss(params['loss_function'])
        return self._fit(X, y, cat_features, text_features, embedding_features, None, sample_weight, None, None, None, None, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

    def predict(self, data, prediction_type=None, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, task_type='CPU'):
        """
        Predict with data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        prediction_type : string, optional (default='RawFormulaVal')
            Can be:
            - 'RawFormulaVal' : return raw formula value.
            - 'Exponent' : return Exponent of raw formula value.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction :
            If data is for a single object, the return value is single float formula return value
            otherwise one-dimensional numpy.ndarray of formula return values for each object.
        """
        if prediction_type is None:
            prediction_type = self._get_default_prediction_type()
        return self._predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose, 'predict', task_type)

    def staged_predict(self, data, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, verbose=None):
        """
        Predict target at each stage for data.

        Parameters
        ----------
        data : catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series
                or catboost.FeaturesData
            Data to apply model on.
            If data is a simple list (not list of lists) or a one-dimensional numpy.ndarray it is interpreted
            as a list of features for a single object.

        ntree_start: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        ntree_end: int, optional (default=0)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).
            If value equals to 0 this parameter is ignored and ntree_end equal to tree_count_.

        eval_period: int, optional (default=1)
            Model is applied on the interval [ntree_start, ntree_end) with the step eval_period (zero-based indexing).

        thread_count : int (default=-1)
            The number of threads to use when applying the model.
            Allows you to optimize the speed of execution. This parameter doesn't affect results.
            If -1, then the number of threads is set to the number of CPU cores.

        verbose : bool
            If True, writes the evaluation metric measured set to stderr.

        Returns
        -------
        prediction : generator for each iteration that generates:
            If data is for a single object, the return value is single float formula return value
            otherwise one-dimensional numpy.ndarray of formula return values for each object.
        """
        return self._staged_predict(data, prediction_type, ntree_start, ntree_end, eval_period, thread_count, verbose, 'staged_predict')

    def score(self, X, y=None):
        """
        Calculate R^2.

        Parameters
        ----------
        X : catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
            Data to apply model on.
        y : list or numpy.ndarray
            True labels.

        Returns
        -------
        R^2 : float
        """
        if isinstance(X, Pool):
            if y is not None:
                raise CatBoostError('Wrong initializing y: X is catboost.Pool object, y must be initialized inside catboost.Pool.')
            y = X.get_label()
            if y is None:
                raise CatBoostError('Label in X has not initialized.')
        elif y is None:
            raise CatBoostError('y should be specified.')
        y = np.array(y, dtype=np.float64)
        predictions = self._predict(X, prediction_type=self._get_default_prediction_type(), ntree_start=0, ntree_end=0, thread_count=-1, verbose=None, parent_method_name='score')
        loss = self._object._get_loss_function_name()
        if loss == 'RMSEWithUncertainty':
            predictions = predictions[:, 0]
        if predictions.size != y.size:
            msg = 'labels and predictions should have same size. But y.size={} != preds.size={}'
            raise CatBoostError(msg.format(y.size, predictions.size))
        y = y.reshape(predictions.shape)
        total_sum_of_squares = np.sum((y - y.mean(axis=0)) ** 2)
        residual_sum_of_squares = np.sum((y - predictions) ** 2)
        return 1 - residual_sum_of_squares / total_sum_of_squares

    @staticmethod
    def _check_is_compatible_loss(loss_function):
        is_regression = CatBoost._is_regression_objective(loss_function) or CatBoost._is_multiregression_objective(loss_function) or CatBoost._is_survivalregression_objective(loss_function)
        if isinstance(loss_function, str) and (not is_regression):
            raise CatBoostError("Invalid loss_function='{}': for regressor use RMSE, MultiRMSE, SurvivalAft, MAE, Quantile, LogLinQuantile, Poisson, MAPE, Lq or custom objective object".format(loss_function))

    def _get_default_prediction_type(self):
        params = deepcopy(self._init_params)
        _process_synonyms(params)
        loss_function = params.get('loss_function')
        if loss_function and isinstance(loss_function, str):
            if loss_function.startswith('Poisson') or loss_function.startswith('Tweedie'):
                return 'Exponent'
            if loss_function == 'RMSEWithUncertainty':
                return 'RMSEWithUncertainty'
        return 'RawFormulaVal'