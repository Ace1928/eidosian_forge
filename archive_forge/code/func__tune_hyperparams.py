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
def _tune_hyperparams(self, param_grid, X, y=None, cv=3, n_iter=10, partition_random_seed=0, calc_cv_statistics=True, search_by_train_test_split=True, refit=True, shuffle=True, stratified=None, train_size=0.8, verbose=1, plot=False, plot_file=None, log_cout=None, log_cerr=None):
    if refit and self.is_fitted():
        raise CatBoostError("Model was fitted before hyperparameters tuning. You can't change hyperparameters of fitted model.")
    with log_fixup(log_cout, log_cerr):
        currently_not_supported_params = {'ignored_features', 'input_borders', 'loss_function', 'eval_metric'}
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        for grid_num, grid in enumerate(param_grid):
            _process_synonyms_groups(grid)
            grid = _params_type_cast(grid)
            for param in currently_not_supported_params:
                if param in grid:
                    raise CatBoostError("Parameter '{}' currently is not supported in hyperparaneter search".format(param))
        if X is None:
            raise CatBoostError('X must not be None')
        if y is None and (not isinstance(X, PATH_TYPES + (Pool,))):
            raise CatBoostError('y may be None only when X is an instance of catboost.Pool, str or pathlib.Path')
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or a list ({!r})'.format(param_grid))
        train_params = self._prepare_train_params(X=X, y=y)
        params = train_params['params']
        custom_folds = None
        fold_count = 0
        if isinstance(cv, INTEGER_TYPES):
            fold_count = cv
            loss_function = params.get('loss_function', None)
            if stratified is None:
                stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)
        else:
            if not hasattr(cv, '__iter__') and (not hasattr(cv, 'split')):
                raise AttributeError('cv should be one of possible things:\n- None, to use the default 3-fold cross validation,\n- integer, to specify the number of folds in a (Stratified)KFold\n- one of the scikit-learn splitter classes (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)\n- An iterable yielding (train, test) splits as arrays of indices')
            custom_folds = cv
            shuffle = False
        if stratified is None:
            loss_function = params.get('loss_function', None)
            stratified = isinstance(loss_function, STRING_TYPES) and is_cv_stratified_objective(loss_function)
        with plot_wrapper(plot, plot_file, 'Hyperparameters search plot', [_get_train_dir(params)]):
            cv_result = self._object._tune_hyperparams(param_grid, train_params['train_pool'], params, n_iter, fold_count, partition_random_seed, shuffle, stratified, train_size, search_by_train_test_split, calc_cv_statistics, custom_folds, verbose)
        if refit:
            assert not self.is_fitted()
            self.set_params(**cv_result['params'])
            self.fit(X, y, silent=True)
    return cv_result