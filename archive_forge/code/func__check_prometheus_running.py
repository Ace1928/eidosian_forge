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
def _check_prometheus_running(self):
    from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
    if self._prometheus_ran_before:
        return
    prometheus_running = False
    try:
        resp = requests.get(f'{self._dashboard_url_base}/api/prometheus_health')
        if resp.status_code == 200:
            json = resp.json()
            prometheus_running = json['result'] is True
    except Exception:
        pass
    record_extra_usage_tag(TagKey.DASHBOARD_METRICS_PROMETHEUS_ENABLED, str(prometheus_running))
    if prometheus_running:
        self._prometheus_ran_before = True