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
def _history_define_metric(self, hkey: str) -> Optional[MetricRecord]:
    """Check for hkey match in glob metrics and return the defined metric."""
    if hkey.startswith('_'):
        return None
    for k, mglob in self._metric_globs.items():
        if k.endswith('*'):
            if hkey.startswith(k[:-1]):
                m = MetricRecord()
                m.CopyFrom(mglob)
                m.ClearField('glob_name')
                m.options.defined = False
                m.name = hkey
                return m
    return None