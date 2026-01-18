import asyncio
import datetime
import json
import logging
import os
import socket
import sys
import traceback
import warnings
import psutil
from typing import List, Optional, Tuple
from collections import defaultdict
import ray
import ray._private.services
import ray._private.utils
from ray.dashboard.consts import (
from ray.dashboard.modules.reporter.profile_manager import CpuProfilingManager
import ray.dashboard.modules.reporter.reporter_consts as reporter_consts
import ray.dashboard.utils as dashboard_utils
from opencensus.stats import stats as stats_module
import ray._private.prometheus_exporter as prometheus_exporter
from prometheus_client.core import REGISTRY
from ray._private.metrics_agent import Gauge, MetricsAgent, Record
from ray._private.ray_constants import DEBUG_AUTOSCALING_STATUS
from ray.core.generated import reporter_pb2, reporter_pb2_grpc
from ray.util.debug import log_once
from ray.dashboard import k8s_utils
from ray._raylet import WorkerID
def _get_raylet_proc(self):
    try:
        if not self._raylet_proc:
            curr_proc = psutil.Process()
            self._raylet_proc = curr_proc.parent()
        if self._raylet_proc is not None:
            if self._raylet_proc.pid == 1:
                return None
            if self._raylet_proc.status() == psutil.STATUS_ZOMBIE:
                return None
        return self._raylet_proc
    except (psutil.AccessDenied, ProcessLookupError):
        pass
    return None