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
def _save_summary(self, summary_dict: SummaryDict, flush: bool=False) -> None:
    summary = SummaryRecord()
    for k, v in summary_dict.items():
        update = summary.update.add()
        update.key = k
        update.value_json = json.dumps(v)
    if flush:
        record = Record(summary=summary)
        self._dispatch_record(record)
    elif not self._settings._offline:
        summary_record = SummaryRecordRequest(summary=summary)
        request_record = self._interface._make_request(summary_record=summary_record)
        self._dispatch_record(request_record)