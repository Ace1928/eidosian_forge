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
def _get_trial_location(trial: Trial, result: dict) -> _Location:
    node_ip, pid = (result.get(NODE_IP, None), result.get(PID, None))
    if node_ip and pid:
        location = _Location(node_ip, pid)
    else:
        location = trial.temporary_state.location
    return location