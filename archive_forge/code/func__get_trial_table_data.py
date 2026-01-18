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
def _get_trial_table_data(trials: List[Trial], param_keys: List[str], metric_keys: List[str], all_rows: bool=False, wrap_headers: bool=False) -> _TrialTableData:
    """Generate a table showing the current progress of tuning trials.

    Args:
        trials: List of trials for which progress is to be shown.
        param_keys: Ordered list of parameters to be displayed in the table.
        metric_keys: Ordered list of metrics to be displayed in the table.
            Including both default and user defined.
            Will only be shown if at least one trial is having the key.
        all_rows: Force to show all rows.
        wrap_headers: If True, header columns can be wrapped with ``
``.

    Returns:
        Trial table data, including header and trial table per each status.
    """
    max_trial_num_to_show = 20
    max_column_length = 20
    trials_by_state = _get_trials_by_state(trials)
    metric_keys = [k for k in metric_keys if any((unflattened_lookup(k, t.last_result, default=None) is not None for t in trials))]
    formatted_metric_columns = [_max_len(k, max_len=max_column_length, wrap=wrap_headers) for k in metric_keys]
    formatted_param_columns = [_max_len(k, max_len=max_column_length, wrap=wrap_headers) for k in param_keys]
    metric_header = [DEFAULT_COLUMNS[metric] if metric in DEFAULT_COLUMNS else formatted for metric, formatted in zip(metric_keys, formatted_metric_columns)]
    param_header = formatted_param_columns
    header = ['Trial name', 'status'] + param_header + metric_header
    trial_data = list()
    for t_status in ORDER:
        trial_data_per_status = _get_trial_table_data_per_status(t_status, trials_by_state[t_status], param_keys=param_keys, metric_keys=metric_keys, force_max_rows=not all_rows and len(trials) > max_trial_num_to_show)
        if trial_data_per_status:
            trial_data.append(trial_data_per_status)
    return _TrialTableData(header, trial_data)