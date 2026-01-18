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
def _get_gpu_usage():
    global enable_gpu_usage_check
    if gpustat is None or not enable_gpu_usage_check:
        return []
    gpu_utilizations = []
    gpus = []
    try:
        gpus = gpustat.new_query().gpus
    except Exception as e:
        logger.debug(f'gpustat failed to retrieve GPU information: {e}')
        if type(e).__name__ == 'NVMLError_DriverNotLoaded':
            enable_gpu_usage_check = False
    for gpu in gpus:
        gpu_data = {'_'.join(key.split('.')): val for key, val in gpu.entry.items()}
        gpu_utilizations.append(gpu_data)
    return gpu_utilizations