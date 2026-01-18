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
def _set_trial_status(self, trial: Trial, status: str):
    """Set trial to a specific status.

        This will keep track of trials with specific statuses in sets.

        For PENDING and PAUSED trials we also keep a list of trials to be able
        to retain FIFO ordering. See ``_maybe_add_actors`` for details.

        Lastly we also keep a mapping from resources to pending/paused trials
        to be able to efficiently start trials for cached actors.
        """
    current_status = trial.status
    if current_status == status:
        logger.debug(f'Trial {trial} already has status {status}. Skipping update.')
        return
    status_str_map = {Trial.PENDING: self._pending_trials, Trial.RUNNING: self._running_trials, Trial.PAUSED: self._paused_trials, Trial.TERMINATED: self._stopped_trials, Trial.ERROR: self._failed_trials}
    logger.debug(f'Setting status for trial {trial} from {current_status} to {status}')
    assert trial in status_str_map[current_status], (trial, current_status)
    assert trial not in status_str_map[status], (trial, status)
    status_str_map[current_status].remove(trial)
    status_str_map[status].add(trial)
    if status == Trial.PENDING:
        self._pending_trials_list.append(trial)
        self._resources_to_pending_trials[trial.placement_group_factory].add(trial)
    else:
        self._resources_to_pending_trials[trial.placement_group_factory].discard(trial)
    trial.set_status(status)