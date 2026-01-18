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
def _schedule_trial_stop(self, trial: Trial, exception: Optional[Exception]=None):
    if trial.status == Trial.ERROR:
        logger.debug(f'Not requesting trial STOP as it is ERROR already: {trial}')
        return
    logger.debug(f'Requesting to STOP actor for trial {trial}')
    if trial.is_saving:
        logger.debug(f'Trial {trial} is currently saving/pausing. Scheduling STOP after save resolved.')
        self._cached_trial_decisions[trial.trial_id] = TrialScheduler.STOP
    trial.temporary_state.saving_to = None
    trial.temporary_state.restoring_from = None
    self._set_trial_status(trial, Trial.ERROR if exception else Trial.TERMINATED)
    trial.set_location(_Location())
    if trial not in self._trial_to_actor:
        logger.debug(f'Will not STOP trial actor as it is not live: {trial}')
        return
    tracked_actor = self._trial_to_actor[trial]
    self._actor_manager.clear_actor_task_futures(tracked_actor=tracked_actor)
    self._mark_trial_to_checkpoint(trial)
    if not exception and self._maybe_cache_trial_actor(trial):
        return
    logger.debug(f'Terminating actor for trial {trial}: {tracked_actor}')
    tracked_actor = self._trial_to_actor.pop(trial)
    self._actor_to_trial.pop(tracked_actor)
    trial.set_ray_actor(None)
    self._remove_actor(tracked_actor=tracked_actor)