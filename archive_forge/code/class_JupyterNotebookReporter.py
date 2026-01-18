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
class JupyterNotebookReporter(TuneReporterBase, RemoteReporterMixin):
    """Jupyter notebook-friendly Reporter that can update display in-place.

    Args:
        overwrite: Flag for overwriting the cell contents before initialization.
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

    def __init__(self, *, overwrite: bool=True, metric_columns: Optional[Union[List[str], Dict[str, str]]]=None, parameter_columns: Optional[Union[List[str], Dict[str, str]]]=None, total_samples: Optional[int]=None, max_progress_rows: int=20, max_error_rows: int=20, max_column_length: int=20, max_report_frequency: int=5, infer_limit: int=3, print_intermediate_tables: Optional[bool]=None, metric: Optional[str]=None, mode: Optional[str]=None, sort_by_metric: bool=False):
        super(JupyterNotebookReporter, self).__init__(metric_columns=metric_columns, parameter_columns=parameter_columns, total_samples=total_samples, max_progress_rows=max_progress_rows, max_error_rows=max_error_rows, max_column_length=max_column_length, max_report_frequency=max_report_frequency, infer_limit=infer_limit, print_intermediate_tables=print_intermediate_tables, metric=metric, mode=mode, sort_by_metric=sort_by_metric)
        if not IS_NOTEBOOK:
            warnings.warn('You are using the `JupyterNotebookReporter`, but not IPython/Jupyter-compatible environment was detected. If this leads to unformatted output (e.g. like <IPython.core.display.HTML object>), consider passing a `CLIReporter` as the `progress_reporter` argument to `train.RunConfig()` instead.')
        self._overwrite = overwrite
        self._display_handle = None
        self.display('')

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        progress = self._progress_html(trials, done, *sys_info)
        if self.output_queue is not None:
            self.output_queue.put(progress)
        else:
            self.display(progress)

    def display(self, string: str) -> None:
        from IPython.display import HTML, clear_output, display
        if not self._display_handle:
            if self._overwrite:
                clear_output(wait=True)
            self._display_handle = display(HTML(string), display_id=True)
        else:
            self._display_handle.update(HTML(string))

    def _progress_html(self, trials: List[Trial], done: bool, *sys_info) -> str:
        """Generate an HTML-formatted progress update.

        Args:
            trials: List of trials for which progress should be
                displayed
            done: True if the trials are finished, False otherwise
            *sys_info: System information to be displayed

        Returns:
            Progress update to be rendered in a notebook, including HTML
                tables and formatted error messages. Includes
                - Duration of the tune job
                - Memory consumption
                - Trial progress table, with information about each experiment
        """
        if not self._metrics_override:
            user_metrics = self._infer_user_metrics(trials, self._infer_limit)
            self._metric_columns.update(user_metrics)
        current_time, running_for = _get_time_str(self._start_time, time.time())
        used_gb, total_gb, memory_message = _get_memory_usage()
        status_table = tabulate([('Current time:', current_time), ('Running for:', running_for), ('Memory:', f'{used_gb}/{total_gb} GiB')], tablefmt='html')
        trial_progress_data = _trial_progress_table(trials=trials, metric_columns=self._metric_columns, parameter_columns=self._parameter_columns, fmt='html', max_rows=None if done else self._max_progress_rows, metric=self._metric, mode=self._mode, sort_by_metric=self._sort_by_metric, max_column_length=self._max_column_length)
        trial_progress = trial_progress_data[0]
        trial_progress_messages = trial_progress_data[1:]
        trial_errors = _trial_errors_str(trials, fmt='html', max_rows=None if done else self._max_error_rows)
        if any([memory_message, trial_progress_messages, trial_errors]):
            msg = Template('tune_status_messages.html.j2').render(memory_message=memory_message, trial_progress_messages=trial_progress_messages, trial_errors=trial_errors)
        else:
            msg = None
        return Template('tune_status.html.j2').render(status_table=status_table, sys_info_message=_generate_sys_info_str(*sys_info), trial_progress=trial_progress, messages=msg)