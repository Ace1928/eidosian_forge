import functools
import logging
import os
import platform
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, Type
import ray
from ray.air._internal.session import _get_session
from ray.air._internal.util import RunnerThread, StartTraceback
from ray.air.constants import (
from ray.data import Dataset
from ray.train import Checkpoint
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.storage import StorageContext
from ray.train.constants import (
from ray.train.error import SessionMisuseError
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.debug import log_once
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
class _FutureTrainingResult:
    """A future that will be resolved to a `_TrainingResult`.

    This is needed for specific schedulers such as PBT that schedule saves.

    This wrapper should be removed after refactoring PBT to not schedule saves anymore.
    """

    def __init__(self, future: ray.ObjectRef):
        self.future = future

    def resolve(self, block: bool=True) -> Optional['_TrainingResult']:
        """Resolve into ``_TrainingResult``.

        This will return None for function trainables if no checkpoint has been
        saved before.
        """
        if block:
            timeout = None
        else:
            timeout = 1e-09
        try:
            return ray.get(self.future, timeout=timeout)
        except TimeoutError:
            pass
        except Exception as exc:
            logger.error(f'Error resolving result: {exc}')