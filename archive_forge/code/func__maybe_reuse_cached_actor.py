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
def _maybe_reuse_cached_actor(self, trial: Trial) -> bool:
    """Maybe reuse a cached actor for a trial.

        If an actor has been scheduled for the trial already,
        this will remove the original actor.
        """
    if trial in self._resetting_trials:
        return True
    resource_request = trial.placement_group_factory
    if not self._actor_cache.has_cached_object(resource_request):
        return False
    cached_actor = self._actor_cache.pop_cached_object(resource_request)
    logger.debug(f'Reusing ACTOR for trial {trial}: {cached_actor}')
    if trial in self._trial_to_actor:
        original_actor = self._trial_to_actor.pop(trial)
        self._actor_to_trial.pop(original_actor)
        logger.debug(f'Removing ORIGINAL ACTOR for trial {trial}: {original_actor}')
        self._remove_actor(tracked_actor=original_actor)
    self._trial_to_actor[trial] = cached_actor
    self._actor_to_trial[cached_actor] = trial
    ray_actor = self._actor_manager._live_actors_to_ray_actors_resources[cached_actor][0]
    trial.set_ray_actor(ray_actor)
    self._schedule_trial_reset(trial, trial.config, trial.experiment_tag)
    return True