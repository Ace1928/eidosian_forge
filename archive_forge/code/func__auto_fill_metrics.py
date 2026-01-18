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
def _auto_fill_metrics(self, result: dict) -> dict:
    """Add autofilled metrics and update attributes."""
    current_time = time.time()
    current_datetime = datetime.now()
    if TIME_THIS_ITER_S in result:
        time_this_iter = result[TIME_THIS_ITER_S]
    else:
        time_this_iter = current_time - self.last_report_time
    self.iteration += 1
    self.time_total += time_this_iter
    self.last_report_time = current_time
    auto_filled_metrics = {TIMESTAMP: int(time.mktime(current_datetime.timetuple())), TIME_TOTAL_S: self.time_total, WORKER_PID: os.getpid(), WORKER_HOSTNAME: platform.node(), WORKER_NODE_IP: self.local_ip}
    if not self.detailed_autofilled_metrics:
        auto_filled_metrics = {k: v for k, v in auto_filled_metrics.items() if k not in DETAILED_AUTOFILLED_KEYS}
    result = result.copy()
    result.update(auto_filled_metrics)
    return result