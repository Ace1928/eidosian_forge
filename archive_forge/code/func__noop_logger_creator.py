import copy
import json
import logging
from contextlib import contextmanager
from functools import partial
from numbers import Number
import os
from pathlib import Path
import platform
import re
import time
from typing import Any, Dict, Optional, Sequence, Union, Callable, List, Tuple
import uuid
import ray
from ray.air.constants import (
import ray.cloudpickle as cloudpickle
from ray.exceptions import RayActorError, RayTaskError
from ray.train import Checkpoint, CheckpointConfig
from ray.train.constants import (
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.train._internal.session import _FutureTrainingResult, _TrainingResult
from ray.train._internal.storage import StorageContext
from ray.tune import TuneError
from ray.tune.logger import NoopLogger
from ray.tune.registry import get_trainable_cls, validate_trainable
from ray.tune.result import (
from ray.tune.execution.placement_groups import (
from ray.tune.trainable.metadata import _TrainingRunMetadata
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.tune.utils import date_str, flatten_dict
from ray.util.annotations import DeveloperAPI, Deprecated
from ray._private.utils import binary_to_hex, hex_to_binary
def _noop_logger_creator(config: Dict[str, Any], logdir: str):
    os.environ.setdefault('TUNE_ORIG_WORKING_DIR', os.getcwd())
    os.makedirs(logdir, exist_ok=True)
    if bool(int(os.environ.get(RAY_CHDIR_TO_TRIAL_DIR, '1'))):
        if not ray._private.worker._mode() == ray._private.worker.LOCAL_MODE:
            os.chdir(logdir)
    return NoopLogger(config, logdir)