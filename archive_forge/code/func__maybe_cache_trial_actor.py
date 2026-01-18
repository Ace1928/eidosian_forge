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
def _maybe_cache_trial_actor(self, trial: Trial) -> bool:
    """Cache trial actor for reuse, if needed.

        We will only cache as many actors as are needed to fulfill any pending
        resource requests for actors with the same resource requirements.
        E.g. if we have 6 running trials and 4 additional staged actors, we will only
        cache up to 4 of the running trial actors when they finish.

        One exception is the case when we have no cached actors, yet. In that case,
        we will always cache the actor in this method.

        Later, in `_cleanup_cached_actors`, we will check again if we need this cached
        actor. That method will keep the actor if we don't have any staged trials,
        because we don't know at that point if the next trial might require the same
        resources. But because there is no staged trial, it is safe to keep the actor
        around, as it won't occupy resources needed by another trial until it's staged.
        """
    if not self._reuse_actors:
        return False
    if self._search_alg.is_finished() and (not self._staged_trials):
        logger.debug(f'Not caching actor of trial {trial} as the search is over and no more trials are staged.')
        return False
    tracked_actor = self._trial_to_actor[trial]
    if not self._actor_manager.is_actor_started(tracked_actor) or self._actor_manager.is_actor_failed(tracked_actor) or tracked_actor not in self._started_actors:
        logger.debug(f'Not caching actor of trial {trial} as it has not been started, yet: {tracked_actor}')
        return False
    if not self._actor_cache.cache_object(trial.placement_group_factory, tracked_actor):
        logger.debug(f'Could not cache actor of trial {trial} for reuse, as there are no pending trials requiring its resources.')
        return False
    logger.debug(f'Caching actor of trial {trial} for re-use: {tracked_actor}')
    tracked_actor = self._trial_to_actor.pop(trial)
    self._actor_to_trial.pop(tracked_actor)
    trial.set_ray_actor(None)
    return True