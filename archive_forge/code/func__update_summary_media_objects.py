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
def _update_summary_media_objects(self, v: Dict[str, Any]) -> Dict[str, Any]:
    for nk, nv in v.items():
        if isinstance(nv, dict) and handler_util.metric_is_wandb_dict(nv) and ('_latest_artifact_path' in nv) and ('artifact_path' in nv):
            nv['artifact_path'] = nv['_latest_artifact_path']
            v[nk] = nv
    return v