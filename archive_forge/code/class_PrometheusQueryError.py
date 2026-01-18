import asyncio
import aiohttp
import logging
import os
import shutil
from typing import Optional
import psutil
from urllib.parse import quote
from ray.dashboard.modules.metrics.grafana_dashboard_factory import (
from ray.dashboard.modules.metrics.grafana_datasource_template import (
from ray.dashboard.modules.metrics.grafana_dashboard_provisioning_template import (
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.consts import AVAILABLE_COMPONENT_NAMES_FOR_METRICS
class PrometheusQueryError(Exception):

    def __init__(self, status, message):
        self.message = f'Error fetching data from prometheus. status: {status}, message: {message}'
        super().__init__(self.message)