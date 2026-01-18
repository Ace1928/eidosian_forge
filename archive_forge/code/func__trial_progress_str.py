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
def _trial_progress_str(trials: List[Trial], metric_columns: Union[List[str], Dict[str, str]], parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: int=0, force_table: bool=False, fmt: str='psql', max_rows: Optional[int]=None, max_column_length: int=20, done: bool=False, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
    """Returns a human readable message for printing to the console.

    This contains a table where each row represents a trial, its parameters
    and the current values of its metrics.

    Args:
        trials: List of trials to get progress string for.
        metric_columns: Names of metrics to include.
            If this is a dict, the keys are metric names and the values are
            the names to use in the message. If this is a list, the metric
            name is used in the message directly.
        parameter_columns: Names of parameters to
            include. If this is a dict, the keys are parameter names and the
            values are the names to use in the message. If this is a list,
            the parameter name is used in the message directly. If this is
            empty, all parameters are used in the message.
        total_samples: Total number of trials that will be generated.
        force_table: Force printing a table. If False, a table will
            be printed only at the end of the training for verbosity levels
            above `Verbosity.V2_TRIAL_NORM`.
        fmt: Output format (see tablefmt in tabulate API).
        max_rows: Maximum number of rows in the trial table. Defaults to
            unlimited.
        max_column_length: Maximum column length (in characters).
        done: True indicates that the tuning run finished.
        metric: Metric used to sort trials.
        mode: One of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute.
        sort_by_metric: Sort terminated trials by metric in the
            intermediate table. Defaults to False.
    """
    messages = []
    delim = '<br>' if fmt == 'html' else '\n'
    if len(trials) < 1:
        return delim.join(messages)
    num_trials = len(trials)
    trials_by_state = _get_trials_by_state(trials)
    for local_dir in sorted({t.local_experiment_path for t in trials}):
        messages.append('Result logdir: {}'.format(local_dir))
    num_trials_strs = ['{} {}'.format(len(trials_by_state[state]), state) for state in sorted(trials_by_state)]
    if total_samples and total_samples >= sys.maxsize:
        total_samples = 'infinite'
    messages.append('Number of trials: {}{} ({})'.format(num_trials, f'/{total_samples}' if total_samples else '', ', '.join(num_trials_strs)))
    if force_table or (has_verbosity(Verbosity.V2_TRIAL_NORM) and done):
        messages += _trial_progress_table(trials=trials, metric_columns=metric_columns, parameter_columns=parameter_columns, fmt=fmt, max_rows=max_rows, metric=metric, mode=mode, sort_by_metric=sort_by_metric, max_column_length=max_column_length)
    return delim.join(messages)