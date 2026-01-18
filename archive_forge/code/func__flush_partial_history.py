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
def _flush_partial_history(self, step: Optional[int]=None) -> None:
    if not self._partial_history:
        return
    history = HistoryRecord()
    for k, v in self._partial_history.items():
        item = history.item.add()
        item.key = k
        item.value_json = json.dumps(v)
    if step is not None:
        history.step.num = step
    self.handle_history(Record(history=history))
    self._partial_history = {}