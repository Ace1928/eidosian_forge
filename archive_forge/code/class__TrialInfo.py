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
class _TrialInfo:
    """Serializable struct for holding information for a Trial.

    Attributes:
        trial_name: String name of the current trial.
        trial_id: trial_id of the trial
        trial_resources: resources used by trial.
    """

    def __init__(self, trial: 'Trial'):
        self._trial_name = str(trial)
        self._trial_id = trial.trial_id
        self._trial_resources = trial.placement_group_factory
        self._experiment_name = trial.experiment_dir_name

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def trial_name(self):
        return self._trial_name

    @property
    def trial_id(self):
        return self._trial_id

    @property
    def trial_resources(self) -> PlacementGroupFactory:
        return self._trial_resources

    @trial_resources.setter
    def trial_resources(self, new_resources: PlacementGroupFactory):
        self._trial_resources = new_resources