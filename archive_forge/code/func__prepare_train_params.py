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
def _prepare_train_params(self, X=None, y=None, cat_features=None, text_features=None, embedding_features=None, pairs=None, sample_weight=None, group_id=None, group_weight=None, subgroup_id=None, pairs_weight=None, baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=None, plot_file=None, column_description=None, verbose_eval=None, metric_period=None, silent=None, early_stopping_rounds=None, save_snapshot=None, snapshot_file=None, snapshot_interval=None, init_model=None, callbacks=None):
    params = deepcopy(self._init_params)
    if params is None:
        params = {}
    _process_synonyms(params)
    if isinstance(X, FeaturesData):
        warnings.warn('FeaturesData is deprecated for using in fit function and soon will not be supported. If you want to use FeaturesData, please pass it to Pool initialization and use Pool in fit')
    cat_features = _process_feature_indices(cat_features, X, params, 'cat_features')
    text_features = _process_feature_indices(text_features, X, params, 'text_features')
    embedding_features = _process_feature_indices(embedding_features, X, params, 'embedding_features')
    train_pool = _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description)
    if train_pool.is_empty_:
        raise CatBoostError('X is empty.')
    allow_clear_pool = not isinstance(X, Pool)
    params['loss_function'] = _get_loss_function_for_train(params, getattr(self, '_estimator_type', None), train_pool)
    metric_period, verbose, logging_level = _process_verbose(metric_period, verbose, logging_level, verbose_eval, silent)
    if metric_period is not None:
        params['metric_period'] = metric_period
    if logging_level is not None:
        params['logging_level'] = logging_level
    if verbose is not None:
        params['verbose'] = verbose
    if use_best_model is not None:
        params['use_best_model'] = use_best_model
    if early_stopping_rounds is not None:
        params['od_type'] = 'Iter'
        params['od_wait'] = early_stopping_rounds
        if 'od_pval' in params:
            del params['od_pval']
    if save_snapshot is not None:
        params['save_snapshot'] = save_snapshot
    if snapshot_file is not None:
        params['snapshot_file'] = snapshot_file
    if snapshot_interval is not None:
        params['snapshot_interval'] = snapshot_interval
    if callbacks is not None:
        params['callbacks'] = _TrainCallbacksWrapper(callbacks)
    _check_param_types(params)
    params = _params_type_cast(params)
    _check_train_params(params)
    if params.get('eval_fraction', 0.0) != 0.0:
        if eval_set is not None:
            raise CatBoostError('Both eval_fraction and eval_set specified')
        train_pool, eval_set = self._dataset_train_eval_split(train_pool, params, save_eval_pool=True)
    eval_set_list = eval_set if isinstance(eval_set, list) else [eval_set]
    eval_sets = []
    eval_total_row_count = 0
    for eval_set in eval_set_list:
        if isinstance(eval_set, Pool):
            eval_sets.append(eval_set)
            eval_total_row_count += eval_sets[-1].num_row()
            if eval_sets[-1].num_row() == 0:
                raise CatBoostError("Empty 'eval_set' in Pool")
        elif isinstance(eval_set, PATH_TYPES):
            eval_sets.append(Pool(eval_set, column_description=column_description))
            eval_total_row_count += eval_sets[-1].num_row()
            if eval_sets[-1].num_row() == 0:
                raise CatBoostError("Empty 'eval_set' in file {}".format(eval_set))
        elif isinstance(eval_set, tuple):
            if len(eval_set) != 2:
                raise CatBoostError("Invalid shape of 'eval_set': {}, must be (X, y).".format(str(tuple((type(_) for _ in eval_set)))))
            if eval_set[0] is None or eval_set[1] is None:
                raise CatBoostError("'eval_set' tuple contains at least one None value")
            eval_sets.append(Pool(eval_set[0], eval_set[1], cat_features=train_pool.get_cat_feature_indices(), text_features=train_pool.get_text_feature_indices(), embedding_features=train_pool.get_embedding_feature_indices()))
            eval_total_row_count += eval_sets[-1].num_row()
            if eval_sets[-1].num_row() == 0:
                raise CatBoostError("Empty 'eval_set' in tuple")
        elif eval_set is None:
            if len(eval_set_list) > 1:
                raise CatBoostError('Multiple eval set shall not contain None')
        else:
            raise CatBoostError("Invalid type of 'eval_set': {}, while expected Pool or (X, y) or filename, or list thereof.".format(type(eval_set)))
    if self.get_param('use_best_model') and eval_total_row_count == 0:
        raise CatBoostError("To employ param {'use_best_model': True} provide non-empty 'eval_set'.")
    if init_model is not None and isinstance(init_model, PATH_TYPES):
        try:
            init_model = CatBoost().load_model(init_model)
        except Exception as e:
            raise CatBoostError('Error while loading init_model: {}'.format(e))
    return {'train_pool': train_pool, 'eval_sets': eval_sets, 'params': params, 'allow_clear_pool': allow_clear_pool, 'init_model': init_model}