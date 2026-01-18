import json
import logging
import math
import numbers
import time
from collections import defaultdict
from queue import Queue
from threading import Event
from typing import (
from wandb.proto.wandb_internal_pb2 import (
from ..interface.interface_queue import InterfaceQueue
from ..lib import handler_util, proto_util, tracelog, wburls
from . import context, sample, tb_watcher
from .settings_static import SettingsStatic
from .system.system_monitor import SystemMonitor
def _update_summary_leaf(self, kl: List[str], v: Any, d: Optional[MetricRecord]=None) -> bool:
    has_summary = d and d.HasField('summary')
    if len(kl) == 1:
        copy_key = tuple(kl)
        old_copy = self._metric_copy.get(copy_key)
        if old_copy is None or v != old_copy:
            self._metric_copy[copy_key] = v
            if not has_summary or (d and d.summary.copy):
                self._consolidated_summary[kl[0]] = v
                return True
    if not d:
        return False
    if not has_summary:
        return False
    if not isinstance(v, numbers.Real):
        return False
    if math.isnan(v):
        return False
    float_v = float(v)
    goal_max = None
    if d.goal:
        goal_max = d.goal == d.GOAL_MAXIMIZE
    if self._update_summary_metrics(d.summary, kl=kl, v=v, float_v=float_v, goal_max=goal_max):
        return True
    return False