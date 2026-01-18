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
@staticmethod
def _get_disk_usage():
    if IN_KUBERNETES_POD and (not ENABLE_K8S_DISK_USAGE):
        return {'/': psutil._common.sdiskusage(total=1, used=0, free=1, percent=0.0)}
    if sys.platform == 'win32':
        root = psutil.disk_partitions()[0].mountpoint
    else:
        root = os.sep
    tmp = ray._private.utils.get_user_temp_dir()
    return {'/': psutil.disk_usage(root), tmp: psutil.disk_usage(tmp)}