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
def _update_summary_list(self, kl: List[str], v: Any, d: Optional[MetricRecord]=None) -> bool:
    metric_key = '.'.join([k.replace('.', '\\.') for k in kl])
    d = self._metric_defines.get(metric_key, d)
    if isinstance(v, dict) and (not handler_util.metric_is_wandb_dict(v)):
        updated = False
        for nk, nv in v.items():
            if self._update_summary_list(kl=kl[:] + [nk], v=nv, d=d):
                updated = True
        return updated
    elif isinstance(v, dict) and handler_util.metric_is_wandb_dict(v):
        if '_latest_artifact_path' in v and 'artifact_path' in v:
            v['artifact_path'] = v['_latest_artifact_path']
    updated = self._update_summary_leaf(kl=kl, v=v, d=d)
    return updated