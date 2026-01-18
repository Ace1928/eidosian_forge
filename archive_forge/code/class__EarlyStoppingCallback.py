from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
class _EarlyStoppingCallback:
    """Internal early stopping callable class."""

    def __init__(self, stopping_rounds: int, first_metric_only: bool=False, verbose: bool=True, min_delta: Union[float, List[float]]=0.0) -> None:
        if not isinstance(stopping_rounds, int) or stopping_rounds <= 0:
            raise ValueError(f'stopping_rounds should be an integer and greater than 0. got: {stopping_rounds}')
        self.order = 30
        self.before_iteration = False
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = first_metric_only
        self.verbose = verbose
        self.min_delta = min_delta
        self.enabled = True
        self._reset_storages()

    def _reset_storages(self) -> None:
        self.best_score: List[float] = []
        self.best_iter: List[int] = []
        self.best_score_list: List[_ListOfEvalResultTuples] = []
        self.cmp_op: List[Callable[[float, float], bool]] = []
        self.first_metric = ''

    def _gt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score > best_score + delta

    def _lt_delta(self, curr_score: float, best_score: float, delta: float) -> bool:
        return curr_score < best_score - delta

    def _is_train_set(self, ds_name: str, eval_name: str, env: CallbackEnv) -> bool:
        """Check, by name, if a given Dataset is the training data."""
        if ds_name == 'cv_agg' and eval_name == 'train':
            return True
        if isinstance(env.model, Booster) and ds_name == env.model._train_data_name:
            return True
        return False

    def _init(self, env: CallbackEnv) -> None:
        if env.evaluation_result_list is None or env.evaluation_result_list == []:
            raise ValueError('For early stopping, at least one dataset and eval metric is required for evaluation')
        is_dart = any((env.params.get(alias, '') == 'dart' for alias in _ConfigAliases.get('boosting')))
        if is_dart:
            self.enabled = False
            _log_warning('Early stopping is not available in dart mode')
            return
        if isinstance(env.model, Booster):
            only_train_set = len(env.evaluation_result_list) == 1 and self._is_train_set(ds_name=env.evaluation_result_list[0][0], eval_name=env.evaluation_result_list[0][1].split(' ')[0], env=env)
            if only_train_set:
                self.enabled = False
                _log_warning('Only training set found, disabling early stopping.')
                return
        if self.verbose:
            _log_info(f"Training until validation scores don't improve for {self.stopping_rounds} rounds")
        self._reset_storages()
        n_metrics = len({m[1] for m in env.evaluation_result_list})
        n_datasets = len(env.evaluation_result_list) // n_metrics
        if isinstance(self.min_delta, list):
            if not all((t >= 0 for t in self.min_delta)):
                raise ValueError('Values for early stopping min_delta must be non-negative.')
            if len(self.min_delta) == 0:
                if self.verbose:
                    _log_info('Disabling min_delta for early stopping.')
                deltas = [0.0] * n_datasets * n_metrics
            elif len(self.min_delta) == 1:
                if self.verbose:
                    _log_info(f'Using {self.min_delta[0]} as min_delta for all metrics.')
                deltas = self.min_delta * n_datasets * n_metrics
            else:
                if len(self.min_delta) != n_metrics:
                    raise ValueError('Must provide a single value for min_delta or as many as metrics.')
                if self.first_metric_only and self.verbose:
                    _log_info(f'Using only {self.min_delta[0]} as early stopping min_delta.')
                deltas = self.min_delta * n_datasets
        else:
            if self.min_delta < 0:
                raise ValueError('Early stopping min_delta must be non-negative.')
            if self.min_delta > 0 and n_metrics > 1 and (not self.first_metric_only) and self.verbose:
                _log_info(f'Using {self.min_delta} as min_delta for all metrics.')
            deltas = [self.min_delta] * n_datasets * n_metrics
        self.first_metric = env.evaluation_result_list[0][1].split(' ')[-1]
        for eval_ret, delta in zip(env.evaluation_result_list, deltas):
            self.best_iter.append(0)
            if eval_ret[3]:
                self.best_score.append(float('-inf'))
                self.cmp_op.append(partial(self._gt_delta, delta=delta))
            else:
                self.best_score.append(float('inf'))
                self.cmp_op.append(partial(self._lt_delta, delta=delta))

    def _final_iteration_check(self, env: CallbackEnv, eval_name_splitted: List[str], i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            if self.verbose:
                best_score_str = '\t'.join([_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]])
                _log_info(f'Did not meet early stopping. Best iteration is:\n[{self.best_iter[i] + 1}]\t{best_score_str}')
                if self.first_metric_only:
                    _log_info(f'Evaluated only: {eval_name_splitted[-1]}')
            raise EarlyStopException(self.best_iter[i], self.best_score_list[i])

    def __call__(self, env: CallbackEnv) -> None:
        if env.iteration == env.begin_iteration:
            self._init(env)
        if not self.enabled:
            return
        if env.evaluation_result_list is None:
            raise RuntimeError('early_stopping() callback enabled but no evaluation results found. This is a probably bug in LightGBM. Please report it at https://github.com/microsoft/LightGBM/issues')
        first_time_updating_best_score_list = self.best_score_list == []
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if first_time_updating_best_score_list or self.cmp_op[i](score, self.best_score[i]):
                self.best_score[i] = score
                self.best_iter[i] = env.iteration
                if first_time_updating_best_score_list:
                    self.best_score_list.append(env.evaluation_result_list)
                else:
                    self.best_score_list[i] = env.evaluation_result_list
            eval_name_splitted = env.evaluation_result_list[i][1].split(' ')
            if self.first_metric_only and self.first_metric != eval_name_splitted[-1]:
                continue
            if self._is_train_set(ds_name=env.evaluation_result_list[i][0], eval_name=eval_name_splitted[0], env=env):
                continue
            elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
                if self.verbose:
                    eval_result_str = '\t'.join([_format_eval_result(x, show_stdv=True) for x in self.best_score_list[i]])
                    _log_info(f'Early stopping, best iteration is:\n[{self.best_iter[i] + 1}]\t{eval_result_str}')
                    if self.first_metric_only:
                        _log_info(f'Evaluated only: {eval_name_splitted[-1]}')
                raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
            self._final_iteration_check(env, eval_name_splitted, i)