from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from ray.util.annotations import Deprecated
from xgboost.core import Booster
class TrainingCallback:
    pass