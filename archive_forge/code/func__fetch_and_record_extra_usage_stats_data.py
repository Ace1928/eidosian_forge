import asyncio
import logging
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor
import ray
import ray._private.usage.usage_lib as ray_usage_lib
from ray._private.utils import get_or_create_event_loop
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.utils import async_loop_forever
def _fetch_and_record_extra_usage_stats_data(self):
    logger.debug('Recording dashboard metrics extra telemetry data...')
    self._check_grafana_running()
    self._check_prometheus_running()