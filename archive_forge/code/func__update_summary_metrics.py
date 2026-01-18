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
def _update_summary_metrics(self, s: 'MetricSummary', kl: List[str], v: 'numbers.Real', float_v: float, goal_max: Optional[bool]) -> bool:
    updated = False
    best_key: Optional[Tuple[str, ...]] = None
    if s.none:
        return False
    if s.copy:
        if len(kl) > 1:
            _dict_nested_set(self._consolidated_summary, kl, v)
            return True
    if s.last:
        last_key = tuple(kl + ['last'])
        old_last = self._metric_track.get(last_key)
        if old_last is None or float_v != old_last:
            self._metric_track[last_key] = float_v
            _dict_nested_set(self._consolidated_summary, last_key, v)
            updated = True
    if s.best:
        best_key = tuple(kl + ['best'])
    if s.max or (best_key and goal_max):
        max_key = tuple(kl + ['max'])
        old_max = self._metric_track.get(max_key)
        if old_max is None or float_v > old_max:
            self._metric_track[max_key] = float_v
            if s.max:
                _dict_nested_set(self._consolidated_summary, max_key, v)
                updated = True
            if best_key:
                _dict_nested_set(self._consolidated_summary, best_key, v)
                updated = True
    if s.min or (best_key and (not goal_max)):
        min_key = tuple(kl + ['min'])
        old_min = self._metric_track.get(min_key)
        if old_min is None or float_v < old_min:
            self._metric_track[min_key] = float_v
            if s.min:
                _dict_nested_set(self._consolidated_summary, min_key, v)
                updated = True
            if best_key:
                _dict_nested_set(self._consolidated_summary, best_key, v)
                updated = True
    if s.mean:
        tot_key = tuple(kl + ['tot'])
        num_key = tuple(kl + ['num'])
        avg_key = tuple(kl + ['mean'])
        tot = self._metric_track.get(tot_key, 0.0)
        num = self._metric_track.get(num_key, 0)
        tot += float_v
        num += 1
        self._metric_track[tot_key] = tot
        self._metric_track[num_key] = num
        _dict_nested_set(self._consolidated_summary, avg_key, tot / num)
        updated = True
    return updated