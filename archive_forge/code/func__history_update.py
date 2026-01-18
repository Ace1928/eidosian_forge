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
def _history_update(self, history: HistoryRecord, history_dict: Dict[str, Any]) -> None:
    if history_dict.get('_step') is None:
        self._history_assign_step(history, history_dict)
    update_history: Dict[str, Any] = {}
    if self._metric_defines or self._metric_globs:
        for hkey, hval in history_dict.items():
            self._history_update_list([hkey], hval, history_dict, update_history)
    if update_history:
        history_dict.update(update_history)
        for k, v in update_history.items():
            item = history.item.add()
            item.key = k
            item.value_json = json.dumps(v)