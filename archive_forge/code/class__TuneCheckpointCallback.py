from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from lightgbm.callback import CallbackEnv
from lightgbm.basic import Booster
from ray.util.annotations import Deprecated
class _TuneCheckpointCallback(TuneCallback):

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning('`ray.tune.integration.lightgbm._TuneCheckpointCallback` is deprecated.')