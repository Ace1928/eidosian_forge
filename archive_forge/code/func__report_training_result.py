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
def _report_training_result(self, training_result: _TrainingResult) -> None:
    """Place a training result on the result queue for the main thread to process,
        then block until the main thread signals that training should continue.

        NOTE: This is used internally to report results from Train to Tune
        without persisting checkpoints to storage 2 times.
        `report` is the public API that directly persists to storage, which
        should only be called by user code.
        """
    if training_result.checkpoint:
        self.loaded_checkpoint = training_result.checkpoint
    self.result_queue.put(training_result, block=True)
    self.continue_lock.acquire()
    if self.stop_event.is_set():
        self.stop_event.clear()
        sys.exit(0)