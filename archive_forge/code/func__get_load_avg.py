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
def _get_load_avg(self):
    if sys.platform == 'win32':
        cpu_percent = psutil.cpu_percent()
        load = (cpu_percent, cpu_percent, cpu_percent)
    else:
        load = os.getloadavg()
    if self._cpu_counts[0] > 0:
        per_cpu_load = tuple((round(x / self._cpu_counts[0], 2) for x in load))
    else:
        per_cpu_load = None
    return (load, per_cpu_load)