from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import (
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
def _progress_str(self, trials: List[Trial], done: bool, *sys_info: Dict, fmt: str='psql', delim: str='\n'):
    """Returns full progress string.

        This string contains a progress table and error table. The progress
        table describes the progress of each trial. The error table lists
        the error file, if any, corresponding to each trial. The latter only
        exists if errors have occurred.

        Args:
            trials: Trials to report on.
            done: Whether this is the last progress report attempt.
            fmt: Table format. See `tablefmt` in tabulate API.
            delim: Delimiter between messages.
        """
    if self._sort_by_metric and (self._metric is None or self._mode is None):
        self._sort_by_metric = False
        warnings.warn("Both 'metric' and 'mode' must be set to be able to sort by metric. No sorting is performed.")
    if not self._metrics_override:
        user_metrics = self._infer_user_metrics(trials, self._infer_limit)
        self._metric_columns.update(user_metrics)
    messages = ['== Status ==', _time_passed_str(self._start_time, time.time()), *sys_info]
    if done:
        max_progress = None
        max_error = None
    else:
        max_progress = self._max_progress_rows
        max_error = self._max_error_rows
    current_best_trial, metric = self._current_best_trial(trials)
    if current_best_trial:
        messages.append(_best_trial_str(current_best_trial, metric, self._parameter_columns))
    if has_verbosity(Verbosity.V1_EXPERIMENT):
        messages.append(_trial_progress_str(trials, metric_columns=self._metric_columns, parameter_columns=self._parameter_columns, total_samples=self._total_samples, force_table=self._print_intermediate_tables, fmt=fmt, max_rows=max_progress, max_column_length=self._max_column_length, done=done, metric=self._metric, mode=self._mode, sort_by_metric=self._sort_by_metric))
        messages.append(_trial_errors_str(trials, fmt=fmt, max_rows=max_error))
    return delim.join(messages) + delim