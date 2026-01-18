import abc
import copy
import datetime
import logging
import os
import signal
import sys
import threading
import time
import warnings
from typing import (
import ray
from ray.air._internal import usage as air_usage
from ray.air._internal.usage import AirEntrypoint
from ray.air.util.node import _force_on_current_node
from ray.train import CheckpointConfig, SyncConfig
from ray.train.constants import RAY_CHDIR_TO_TRIAL_DIR, _DEPRECATED_VALUE
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.callback import Callback
from ray.tune.error import TuneError
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Experiment, _convert_to_experiment_list
from ray.tune.experimental.output import (
from ray.tune.impl.placeholder import create_resolvers_map, inject_placeholders
from ray.tune.logger import TBXLoggerCallback
from ray.tune.progress_reporter import (
from ray.tune.registry import get_trainable_cls, is_function_trainable
from ray.tune.schedulers import (
from ray.tune.schedulers.util import (
from ray.tune.stopper import Stopper
from ray.tune.search import (
from ray.tune.search.util import (
from ray.tune.search.variant_generator import _has_unresolved_values
from ray.tune.trainable import Trainable
from ray.tune.experiment import Trial
from ray.tune.utils.callback import _create_default_callbacks
from ray.tune.utils.log import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import PublicAPI
from ray.util.queue import Queue
def _ray_auto_init(entrypoint: str):
    """Initialize Ray unless already configured."""
    if os.environ.get('TUNE_DISABLE_AUTO_INIT') == '1':
        logger.info("'TUNE_DISABLE_AUTO_INIT=1' detected.")
    elif not ray.is_initialized():
        ray.init()
        logger.info(f'Initializing Ray automatically. For cluster usage or custom Ray initialization, call `ray.init(...)` before `{entrypoint}`.')