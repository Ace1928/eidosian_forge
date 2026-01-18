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
@PublicAPI
class CLIReporter(TuneReporterBase):
    """Command-line reporter

    Args:
        metric_columns: Names of metrics to
            include in progress table. If this is a dict, the keys should
            be metric names and the values should be the displayed names.
            If this is a list, the metric name is used directly.
        parameter_columns: Names of parameters to
            include in progress table. If this is a dict, the keys should
            be parameter names and the values should be the displayed names.
            If this is a list, the parameter name is used directly. If empty,
            defaults to all available parameters.
        max_progress_rows: Maximum number of rows to print
            in the progress table. The progress table describes the
            progress of each trial. Defaults to 20.
        max_error_rows: Maximum number of rows to print in the
            error table. The error table lists the error file, if any,
            corresponding to each trial. Defaults to 20.
        max_column_length: Maximum column length (in characters). Column
            headers and values longer than this will be abbreviated.
        max_report_frequency: Maximum report frequency in seconds.
            Defaults to 5s.
        infer_limit: Maximum number of metrics to automatically infer
            from tune results.
        print_intermediate_tables: Print intermediate result
            tables. If None (default), will be set to True for verbosity
            levels above 3, otherwise False. If True, intermediate tables
            will be printed with experiment progress. If False, tables
            will only be printed at then end of the tuning run for verbosity
            levels greater than 2.
        metric: Metric used to determine best current trial.
        mode: One of [min, max]. Determines whether objective is
            minimizing or maximizing the metric attribute.
        sort_by_metric: Sort terminated trials by metric in the
            intermediate table. Defaults to False.
    """

    def __init__(self, *, metric_columns: Optional[Union[List[str], Dict[str, str]]]=None, parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: Optional[int]=None, max_progress_rows: int=20, max_error_rows: int=20, max_column_length: int=20, max_report_frequency: int=5, infer_limit: int=3, print_intermediate_tables: Optional[bool]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
        super(CLIReporter, self).__init__(metric_columns=metric_columns, parameter_columns=parameter_columns, total_samples=total_samples, max_progress_rows=max_progress_rows, max_error_rows=max_error_rows, max_column_length=max_column_length, max_report_frequency=max_report_frequency, infer_limit=infer_limit, print_intermediate_tables=print_intermediate_tables, metric=metric, mode=mode, sort_by_metric=sort_by_metric)

    def _print(self, msg: str):
        safe_print(msg)

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        self._print(self._progress_str(trials, done, *sys_info))