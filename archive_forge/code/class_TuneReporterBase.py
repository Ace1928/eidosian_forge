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
class TuneReporterBase(ProgressReporter):
    _heartbeat_threshold = AirVerbosity.DEFAULT
    _wrap_headers = False
    _intermediate_result_verbosity = AirVerbosity.VERBOSE
    _start_end_verbosity = AirVerbosity.DEFAULT
    _addressing_tmpl = 'Trial {}'

    def __init__(self, verbosity: AirVerbosity, num_samples: int=0, metric: Optional[str]=None, mode: Optional[str]=None, config: Optional[Dict]=None, progress_metrics: Optional[Union[List[str], List[Dict[str, str]]]]=None):
        self._num_samples = num_samples
        self._metric = metric
        self._mode = mode
        self._inferred_metric = None
        self._inferred_params = _infer_params(config or {})
        super(TuneReporterBase, self).__init__(verbosity=verbosity, progress_metrics=progress_metrics)

    def setup(self, start_time: Optional[float]=None, total_samples: Optional[int]=None, **kwargs):
        super().setup(start_time=start_time)
        self._num_samples = total_samples

    def _get_overall_trial_progress_str(self, trials):
        result = ' | '.join([f'{len(trials)} {status}' for status, trials in _get_trials_by_state(trials).items()])
        return f'Trial status: {result}'

    def _get_heartbeat(self, trials, *sys_args, force_full_output: bool=False) -> Tuple[List[str], _TrialTableData]:
        result = list()
        result.append(self._get_overall_trial_progress_str(trials))
        result.append(self._time_heartbeat_str)
        result.extend(sys_args)
        current_best_trial, metric = _current_best_trial(trials, self._metric, self._mode)
        if current_best_trial:
            result.append(_best_trial_str(current_best_trial, metric))
        if not self._inferred_metric:
            self._inferred_metric = _infer_user_metrics(trials)
        all_metrics = list(DEFAULT_COLUMNS.keys()) + self._inferred_metric
        trial_table_data = _get_trial_table_data(trials, param_keys=self._inferred_params, metric_keys=all_metrics, all_rows=force_full_output, wrap_headers=self._wrap_headers)
        return (result, trial_table_data)

    def _print_heartbeat(self, trials, *sys_args, force: bool=False):
        raise NotImplementedError