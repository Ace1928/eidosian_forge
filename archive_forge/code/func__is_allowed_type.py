import enum
import os
import pickle
import urllib
import warnings
import numpy as np
from numbers import Number
import pyarrow.fs
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import ray
from ray import logger
from ray.air import session
from ray.air._internal import usage as air_usage
from ray.air.util.node import _force_on_current_node
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.experiment import Trial
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray._private.storage import _load_class
from ray.util import PublicAPI
from ray.util.queue import Queue
def _is_allowed_type(obj):
    """Return True if type is allowed for logging to wandb"""
    if isinstance(obj, np.ndarray) and obj.size == 1:
        return isinstance(obj.item(), Number)
    if isinstance(obj, Sequence) and len(obj) > 0:
        return isinstance(obj[0], WBValue)
    return isinstance(obj, (Number, WBValue))