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
def _get_heartbeat(self, trials: List[Trial], force_full_output: bool=False):
    if len(trials) == 0:
        return
    trial = trials[0]
    if trial.status != Trial.RUNNING:
        return ' '.join([f'Training is in {trial.status} status.', self._time_heartbeat_str])
    if not trial.last_result or TRAINING_ITERATION not in trial.last_result:
        iter_num = 1
    else:
        iter_num = trial.last_result[TRAINING_ITERATION] + 1
    return ' '.join([f'Training on iteration {iter_num}.', self._time_heartbeat_str])