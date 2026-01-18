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
def _best_trial_str(trial: Trial, metric: str):
    """Returns a readable message stating the current best trial."""
    val = unflattened_lookup(metric, trial.last_result, default=None)
    config = trial.last_result.get('config', {})
    parameter_columns = list(config.keys())
    params = {p: unflattened_lookup(p, config) for p in parameter_columns}
    return f'Current best trial: {trial.trial_id} with {metric}={val} and params={params}'