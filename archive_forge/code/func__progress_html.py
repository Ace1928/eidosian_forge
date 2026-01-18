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