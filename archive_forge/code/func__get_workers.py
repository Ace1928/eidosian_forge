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
def _get_workers(self):
    raylet_proc = self._get_raylet_proc()
    if raylet_proc is None:
        return []
    else:
        workers = {self._generate_worker_key(proc): proc for proc in raylet_proc.children()}
        keys_to_pop = []
        for key, worker in workers.items():
            if key not in self._workers:
                self._workers[key] = worker
        for key in self._workers:
            if key not in workers:
                keys_to_pop.append(key)
        for k in keys_to_pop:
            self._workers.pop(k)
        self._workers.pop(self._generate_worker_key(self._get_agent_proc()))
        result = []
        for w in self._workers.values():
            try:
                if w.status() == psutil.STATUS_ZOMBIE:
                    continue
            except psutil.NoSuchProcess:
                continue
            result.append(w.as_dict(attrs=['pid', 'create_time', 'cpu_percent', 'cpu_times', 'cmdline', 'memory_info', 'memory_full_info', 'num_fds']))
        return result