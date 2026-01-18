import argparse
import sys
from typing import (
import collections
from dataclasses import dataclass
import datetime
from enum import IntEnum
import logging
import math
import numbers
import numpy as np
import os
import pandas as pd
import textwrap
import time
from ray.air._internal.usage import AirEntrypoint
from ray.train import Checkpoint
from ray.tune.search.sample import Domain
from ray.tune.utils.log import Verbosity
import ray
from ray._private.dict import unflattened_lookup, flatten_dict
from ray._private.thirdparty.tabulate.tabulate import (
from ray.air.constants import TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.result import (
from ray.tune.experiment.trial import Trial
class ProgressReporter(Callback):
    """Periodically prints out status update."""
    _heartbeat_freq = 30
    _heartbeat_threshold = None
    _start_end_verbosity = None
    _intermediate_result_verbosity = None
    _addressing_tmpl = None

    def __init__(self, verbosity: AirVerbosity, progress_metrics: Optional[Union[List[str], List[Dict[str, str]]]]=None):
        """

        Args:
            verbosity: AirVerbosity level.
        """
        self._verbosity = verbosity
        self._start_time = time.time()
        self._last_heartbeat_time = float('-inf')
        self._start_time = time.time()
        self._progress_metrics = progress_metrics
        self._trial_last_printed_results = {}
        self._in_block = None

    @property
    def verbosity(self) -> AirVerbosity:
        return self._verbosity

    def setup(self, start_time: Optional[float]=None, **kwargs):
        self._start_time = start_time

    def _start_block(self, indicator: Any):
        if self._in_block != indicator:
            self._end_block()
        self._in_block = indicator

    def _end_block(self):
        if self._in_block:
            print('')
        self._in_block = None

    def on_experiment_end(self, trials: List['Trial'], **info):
        self._end_block()

    def experiment_started(self, experiment_name: str, experiment_path: str, searcher_str: str, scheduler_str: str, total_num_samples: int, tensorboard_path: Optional[str]=None, **kwargs):
        self._start_block('exp_start')
        print(f'\nView detailed results here: {experiment_path}')
        if tensorboard_path:
            print(f'To visualize your results with TensorBoard, run: `tensorboard --logdir {tensorboard_path}`')

    @property
    def _time_heartbeat_str(self):
        current_time_str, running_time_str = _get_time_str(self._start_time, time.time())
        return f'Current time: {current_time_str}. Total running time: ' + running_time_str

    def print_heartbeat(self, trials, *args, force: bool=False):
        if self._verbosity < self._heartbeat_threshold:
            return
        if force or time.time() - self._last_heartbeat_time >= self._heartbeat_freq:
            self._print_heartbeat(trials, *args, force=force)
            self._last_heartbeat_time = time.time()

    def _print_heartbeat(self, trials, *args, force: bool=False):
        raise NotImplementedError

    def _print_result(self, trial, result: Optional[Dict]=None, force: bool=False):
        """Only print result if a different result has been reported, or force=True"""
        result = result or trial.last_result
        last_result_iter = self._trial_last_printed_results.get(trial.trial_id, -1)
        this_iter = result.get(TRAINING_ITERATION, 0)
        if this_iter != last_result_iter or force:
            _print_dict_as_table(result, header=f'{self._addressing_tmpl.format(trial)} result', include=self._progress_metrics, exclude=BLACKLISTED_KEYS, division=AUTO_RESULT_KEYS)
            self._trial_last_printed_results[trial.trial_id] = this_iter

    def _print_config(self, trial):
        _print_dict_as_table(trial.config, header=f'{self._addressing_tmpl.format(trial)} config')

    def on_trial_result(self, iteration: int, trials: List[Trial], trial: Trial, result: Dict, **info):
        if self.verbosity < self._intermediate_result_verbosity:
            return
        self._start_block(f'trial_{trial}_result_{result[TRAINING_ITERATION]}')
        curr_time_str, running_time_str = _get_time_str(self._start_time, time.time())
        print(f'{self._addressing_tmpl.format(trial)} finished iteration {result[TRAINING_ITERATION]} at {curr_time_str}. Total running time: ' + running_time_str)
        self._print_result(trial, result)

    def on_trial_complete(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        if self.verbosity < self._start_end_verbosity:
            return
        curr_time_str, running_time_str = _get_time_str(self._start_time, time.time())
        finished_iter = 0
        if trial.last_result and TRAINING_ITERATION in trial.last_result:
            finished_iter = trial.last_result[TRAINING_ITERATION]
        self._start_block(f'trial_{trial}_complete')
        print(f'{self._addressing_tmpl.format(trial)} completed after {finished_iter} iterations at {curr_time_str}. Total running time: ' + running_time_str)
        self._print_result(trial)

    def on_trial_error(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        curr_time_str, running_time_str = _get_time_str(self._start_time, time.time())
        finished_iter = 0
        if trial.last_result and TRAINING_ITERATION in trial.last_result:
            finished_iter = trial.last_result[TRAINING_ITERATION]
        self._start_block(f'trial_{trial}_error')
        print(f'{self._addressing_tmpl.format(trial)} errored after {finished_iter} iterations at {curr_time_str}. Total running time: {running_time_str}\nError file: {trial.error_file}')
        self._print_result(trial)

    def on_trial_recover(self, iteration: int, trials: List['Trial'], trial: 'Trial', **info):
        self.on_trial_error(iteration=iteration, trials=trials, trial=trial, **info)

    def on_checkpoint(self, iteration: int, trials: List[Trial], trial: Trial, checkpoint: Checkpoint, **info):
        if self._verbosity < self._intermediate_result_verbosity:
            return
        saved_iter = '?'
        if trial.last_result and TRAINING_ITERATION in trial.last_result:
            saved_iter = trial.last_result[TRAINING_ITERATION]
        self._start_block(f'trial_{trial}_result_{saved_iter}')
        loc = f'({checkpoint.filesystem.type_name}){checkpoint.path}'
        print(f'{self._addressing_tmpl.format(trial)} saved a checkpoint for iteration {saved_iter} at: {loc}')

    def on_trial_start(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        if self.verbosity < self._start_end_verbosity:
            return
        has_config = bool(trial.config)
        self._start_block(f'trial_{trial}_start')
        if has_config:
            print(f'{self._addressing_tmpl.format(trial)} started with configuration:')
            self._print_config(trial)
        else:
            print(f'{self._addressing_tmpl.format(trial)} started without custom configuration.')