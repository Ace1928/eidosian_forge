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
def _check_grafana_running(self):
    from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
    if self._grafana_ran_before:
        return
    grafana_running = False
    try:
        resp = requests.get(f'{self._dashboard_url_base}/api/grafana_health')
        if resp.status_code == 200:
            json = resp.json()
            grafana_running = json['result'] is True and json['data']['grafanaHost'] != 'DISABLED'
    except Exception:
        pass
    record_extra_usage_tag(TagKey.DASHBOARD_METRICS_GRAFANA_ENABLED, str(grafana_running))
    if grafana_running:
        self._grafana_ran_before = True