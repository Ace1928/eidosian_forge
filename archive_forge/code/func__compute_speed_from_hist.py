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
def _compute_speed_from_hist(hist):
    while len(hist) > 7:
        hist.pop(0)
    then, prev_stats = hist[0]
    now, now_stats = hist[-1]
    time_delta = now - then
    return tuple(((y - x) / time_delta for x, y in zip(prev_stats, now_stats)))