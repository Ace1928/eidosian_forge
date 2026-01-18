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
def _cleanup_stopping_actors(self, force_all: bool=False):
    now = time.monotonic()
    if not force_all and now - self._earliest_stopping_actor <= self._actor_cleanup_timeout:
        return
    times = deque(sorted([(timestamp, tracked_actor) for tracked_actor, timestamp in self._stopping_actors.items()], key=lambda item: item[0]))
    while times and (force_all or time.monotonic() - times[0][0] > self._actor_cleanup_timeout):
        if time.monotonic() - times[0][0] < self._actor_force_cleanup_timeout and self._actor_manager.is_actor_started(tracked_actor=times[0][1]):
            self._actor_manager.next(timeout=1)
            continue
        _, tracked_actor = times.popleft()
        if tracked_actor not in self._stopping_actors:
            continue
        if self._actor_manager.is_actor_started(tracked_actor=tracked_actor):
            logger.debug(f'Forcefully killing actor: {tracked_actor}')
            self._actor_manager.remove_actor(tracked_actor=tracked_actor, kill=True)
        self._stopping_actors.pop(tracked_actor)
    if times:
        self._earliest_stopping_actor = times[0][0]
    else:
        self._earliest_stopping_actor = float('inf')