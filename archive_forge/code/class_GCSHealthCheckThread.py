import asyncio
import logging
import os
from pathlib import Path
import threading
from concurrent.futures import Future
from queue import Queue
import ray.dashboard.consts as dashboard_consts
import ray.dashboard.utils as dashboard_utils
import ray.experimental.internal_kv as internal_kv
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray._private import ray_constants
from ray.dashboard.utils import DashboardHeadModule
from ray._raylet import GcsClient, check_health
from ray.dashboard.datacenter import DataOrganizer
from ray.dashboard.utils import async_loop_forever
from ray.dashboard.consts import DASHBOARD_METRIC_PORT
from ray.dashboard.dashboard_metrics import DashboardPrometheusMetrics
from typing import Optional, Set
class GCSHealthCheckThread(threading.Thread):

    def __init__(self, gcs_address: str):
        self.gcs_address = gcs_address
        self.work_queue = Queue()
        super().__init__(daemon=True)

    def run(self) -> None:
        while True:
            future = self.work_queue.get()
            check_result = check_health(self.gcs_address)
            future.set_result(check_result)

    async def check_once(self) -> bool:
        """Ask the thread to perform a health check."""
        assert threading.current_thread != self, "caller shouldn't be from the same thread as GCSHealthCheckThread."
        future = Future()
        self.work_queue.put(future)
        return await asyncio.wrap_future(future)