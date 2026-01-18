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
def _current_best_trial(trials: List[Trial], metric: Optional[str], mode: Optional[str]) -> Tuple[Optional[Trial], Optional[str]]:
    """
    Returns the best trial and the metric key. If anything is empty or None,
    returns a trivial result of None, None.

    Args:
        trials: List of trials.
        metric: Metric that trials are being ranked.
        mode: One of "min" or "max".

    Returns:
         Best trial and the metric key.
    """
    if not trials or not metric or (not mode):
        return (None, None)
    metric_op = 1.0 if mode == 'max' else -1.0
    best_metric = float('-inf')
    best_trial = None
    for t in trials:
        if not t.last_result:
            continue
        metric_value = unflattened_lookup(metric, t.last_result, default=None)
        if pd.isnull(metric_value):
            continue
        if not best_trial or metric_value * metric_op > best_metric:
            best_metric = metric_value * metric_op
            best_trial = t
    return (best_trial, metric)