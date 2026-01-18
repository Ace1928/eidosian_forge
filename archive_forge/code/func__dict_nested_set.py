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
def _dict_nested_set(target: Dict[str, Any], key_list: Sequence[str], v: Any) -> None:
    for k in key_list[:-1]:
        target.setdefault(k, {})
        new_target = target.get(k)
        if TYPE_CHECKING:
            new_target = cast(Dict[str, Any], new_target)
        target = new_target
    target[key_list[-1]] = v