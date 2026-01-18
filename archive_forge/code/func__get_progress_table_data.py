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
def _get_progress_table_data(trials: List[Trial], metric_columns: Union[List[str], Dict[str, str]], parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, max_rows: Optional[int]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False, max_column_length: int=20) -> Tuple[List, List[str], Tuple[bool, str]]:
    """Generate a table showing the current progress of tuning trials.

    Args:
        trials: List of trials for which progress is to be shown.
        metric_columns: Metrics to be displayed in the table.
        parameter_columns: List of parameters to be included in the data
        max_rows: Maximum number of rows to show. If there's overflow, a
            message will be shown to the user indicating that some rows
            are not displayed
        metric: Metric which is being tuned
        mode: Sort the table in descending order if mode is "max";
            ascending otherwise
        sort_by_metric: If true, the table will be sorted by the metric
        max_column_length: Max number of characters in each column

    Returns:
        - Trial data
        - List of column names
        - Overflow tuple:
            - boolean indicating whether the table has rows which are hidden
            - string with info about the overflowing rows
    """
    num_trials = len(trials)
    trials_by_state = _get_trials_by_state(trials)
    if sort_by_metric:
        trials_by_state[Trial.TERMINATED] = sorted(trials_by_state[Trial.TERMINATED], reverse=mode == 'max', key=lambda t: unflattened_lookup(metric, t.last_result, default=None))
    state_tbl_order = [Trial.RUNNING, Trial.PAUSED, Trial.PENDING, Trial.TERMINATED, Trial.ERROR]
    max_rows = max_rows or float('inf')
    if num_trials > max_rows:
        trials_by_state_trunc = _fair_filter_trials(trials_by_state, max_rows, sort_by_metric)
        trials = []
        overflow_strs = []
        for state in state_tbl_order:
            if state not in trials_by_state:
                continue
            trials += trials_by_state_trunc[state]
            num = len(trials_by_state[state]) - len(trials_by_state_trunc[state])
            if num > 0:
                overflow_strs.append('{} {}'.format(num, state))
        overflow = num_trials - max_rows
        overflow_str = ', '.join(overflow_strs)
    else:
        overflow = False
        overflow_str = ''
        trials = []
        for state in state_tbl_order:
            if state not in trials_by_state:
                continue
            trials += trials_by_state[state]
    if isinstance(metric_columns, Mapping):
        metric_keys = list(metric_columns.keys())
    else:
        metric_keys = metric_columns
    metric_keys = [k for k in metric_keys if any((unflattened_lookup(k, t.last_result, default=None) is not None for t in trials))]
    if not parameter_columns:
        parameter_keys = sorted(set().union(*[t.evaluated_params for t in trials]))
    elif isinstance(parameter_columns, Mapping):
        parameter_keys = list(parameter_columns.keys())
    else:
        parameter_keys = parameter_columns
    trial_table = [_get_trial_info(trial, parameter_keys, metric_keys, max_column_length=max_column_length) for trial in trials]
    if isinstance(metric_columns, Mapping):
        formatted_metric_columns = [_max_len(metric_columns[k], max_len=max_column_length, add_addr=False, wrap=True) for k in metric_keys]
    else:
        formatted_metric_columns = [_max_len(k, max_len=max_column_length, add_addr=False, wrap=True) for k in metric_keys]
    if isinstance(parameter_columns, Mapping):
        formatted_parameter_columns = [_max_len(parameter_columns[k], max_len=max_column_length, add_addr=False, wrap=True) for k in parameter_keys]
    else:
        formatted_parameter_columns = [_max_len(k, max_len=max_column_length, add_addr=False, wrap=True) for k in parameter_keys]
    columns = ['Trial name', 'status', 'loc'] + formatted_parameter_columns + formatted_metric_columns
    return (trial_table, columns, (overflow, overflow_str))