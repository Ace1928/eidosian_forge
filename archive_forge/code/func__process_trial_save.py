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
def _process_trial_save(self, trial: Trial, checkpoint_value: Union[ray.ObjectRef, str]):
    """Processes a trial save.

        Acts on the decision cached during the last `_process_trial` call.

        Args:
            trial: Trial being saved.
        """
    logger.debug('Trial %s: Processing trial save.', trial)
    try:
        if not checkpoint_value.checkpoint:
            logger.debug(f'Got empty checkpoint for trial {trial}')
        else:
            try:
                self._callbacks.on_checkpoint(iteration=self._iteration, trials=self._trials, trial=trial, checkpoint=checkpoint_value.checkpoint)
            except Exception:
                logger.warning('Error encountered during processing of callbacks. Ray Train/Tune recently changed the checkpoint interface that is passed to callbacks. If you implemented your own callback with an `on_checkpoint` handler, please review the checkpoint interface and adjust your code accordingly.')
                raise
            trial.on_checkpoint(checkpoint_value)
            self._checkpoint_manager.on_trial_checkpoint(trial)
            self._mark_trial_to_checkpoint(trial)
    except Exception:
        logger.exception('Trial %s: Error handling checkpoint %s', trial, checkpoint_value)
    trial.temporary_state.saving_to = None
    decision = self._cached_trial_decisions.pop(trial.trial_id, None)
    if decision and checkpoint_value:
        self._queue_decision(trial, decision)