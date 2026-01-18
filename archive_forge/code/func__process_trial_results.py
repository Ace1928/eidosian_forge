import copy
import json
import time
import traceback
import uuid
import warnings
from collections import defaultdict, deque
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Set
import logging
import os
import ray
from ray.air import ResourceRequest
from ray.air.constants import TIME_THIS_ITER_S
from ray.air.execution import ResourceManager, PlacementGroupResourceManager
from ray.air.execution._internal import RayActorManager, TrackedActor
from ray.train import CheckpointConfig
from ray.train._internal.session import _FutureTrainingResult
from ray.train._internal.storage import StorageContext
from ray.exceptions import RayActorError, RayTaskError
from ray.tune.error import _AbortTrialExecution, _TuneStopTrialError
from ray.tune.execution.class_cache import _ActorClassCache
from ray.tune.execution.experiment_state import (
from ray.tune.experiment.trial import (
from ray.tune.experiment import Experiment
from ray.tune.execution.insufficient_resources_manager import (
from ray.tune.result import (
from ray.tune.result import TRIAL_INFO, STDOUT_FILE, STDERR_FILE
from ray.tune import TuneError
from ray.tune.callback import Callback, CallbackList
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.stopper import NoopStopper, Stopper
from ray.tune.search import BasicVariantGenerator, SearchAlgorithm
from ray.tune.experiment import Trial
from ray.tune.utils.log import _dedup_logs
from ray.tune.utils.object_cache import _ObjectCache
from ray.tune.utils.resource_updater import _ResourceUpdater
from ray.tune.utils import warn_if_slow, flatten_dict
from ray.tune.utils.log import Verbosity, has_verbosity
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.util.annotations import DeveloperAPI, Deprecated
from ray.util.debug import log_once
def _process_trial_results(self, trial, results):
    logger.debug(f'Processing trial results for trial {trial}: {results}')
    with warn_if_slow('process_trial_results', message='Processing trial results took {duration:.3f} s, which may be a performance bottleneck. Please consider reporting results less frequently to Ray Tune.'):
        for i, result in enumerate(results):
            with warn_if_slow('process_trial_result'):
                decision = self._process_trial_result(trial, result)
            if decision is None:
                if i < len(results) - 1:
                    if log_once('tune_controller_buffer_checkpoint'):
                        logger.warning(f'Trial {trial} has a non-training future scheduled but {len(results) - i} results left to process. This means that a checkpoint was requested, but buffered training was continued before it was saved. Consider using non-buffered training by setting the env variable `TUNE_RESULT_BUFFER_LENGTH=1`.')
            elif decision == TrialScheduler.STOP:
                break