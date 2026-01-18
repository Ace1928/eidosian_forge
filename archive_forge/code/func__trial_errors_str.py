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
def _trial_errors_str(trials: List[Trial], fmt: str='psql', max_rows: Optional[int]=None):
    """Returns a readable message regarding trial errors.

    Args:
        trials: List of trials to get progress string for.
        fmt: Output format (see tablefmt in tabulate API).
        max_rows: Maximum number of rows in the error table. Defaults to
            unlimited.
    """
    messages = []
    failed = [t for t in trials if t.error_file]
    num_failed = len(failed)
    if num_failed > 0:
        messages.append('Number of errored trials: {}'.format(num_failed))
        if num_failed > (max_rows or float('inf')):
            messages.append('Table truncated to {} rows ({} overflow)'.format(max_rows, num_failed - max_rows))
        fail_header = ['Trial name', '# failures', 'error file']
        fail_table_data = [[str(trial), str(trial.run_metadata.num_failures) + ('' if trial.status == Trial.ERROR else '*'), trial.error_file] for trial in failed[:max_rows]]
        messages.append(tabulate(fail_table_data, headers=fail_header, tablefmt=fmt, showindex=False, colalign=('left', 'right', 'left')))
        if any((trial.status == Trial.TERMINATED for trial in failed[:max_rows])):
            messages.append('* The trial terminated successfully after retrying.')
    delim = '<br>' if fmt == 'html' else '\n'
    return delim.join(messages)