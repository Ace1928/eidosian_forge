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
def _trial_progress_table(trials: List[Trial], metric_columns: Union[List[str], Dict[str, str]], parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, fmt: str='psql', max_rows: Optional[int]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False, max_column_length: int=20) -> List[str]:
    """Generate a list of trial progress table messages.

    Args:
        trials: List of trials for which progress is to be shown.
        metric_columns: Metrics to be displayed in the table.
        parameter_columns: List of parameters to be included in the data
        fmt: Format of the table; passed to tabulate as the fmtstr argument
        max_rows: Maximum number of rows to show. If there's overflow, a
            message will be shown to the user indicating that some rows
            are not displayed
        metric: Metric which is being tuned
        mode: Sort the table in descenting order if mode is "max";
            ascending otherwise
        sort_by_metric: If true, the table will be sorted by the metric
        max_column_length: Max number of characters in each column

    Returns:
        Messages to be shown to the user containing progress tables
    """
    data, columns, (overflow, overflow_str) = _get_progress_table_data(trials, metric_columns, parameter_columns, max_rows, metric, mode, sort_by_metric, max_column_length)
    messages = [tabulate(data, headers=columns, tablefmt=fmt, showindex=False)]
    if overflow:
        messages.append(f'... {overflow} more trials not shown ({overflow_str})')
    return messages