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
def _handle_defined_metric(self, record: Record) -> None:
    metric = record.metric
    if metric._control.overwrite:
        self._metric_defines[metric.name].CopyFrom(metric)
    else:
        self._metric_defines[metric.name].MergeFrom(metric)
    metric = self._metric_defines[metric.name]
    if metric.step_metric and metric.step_metric not in self._metric_defines:
        m = MetricRecord(name=metric.step_metric)
        self._metric_defines[metric.step_metric] = m
        mr = Record()
        mr.metric.CopyFrom(m)
        mr.control.local = True
        self._dispatch_record(mr)
    self._dispatch_record(record)