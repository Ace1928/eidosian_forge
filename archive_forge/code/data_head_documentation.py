import json
import os
from enum import Enum
import aiohttp
from aiohttp.web import Request, Response
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.modules.metrics.metrics_head import (
from urllib.parse import quote
import ray
import logging
Enum to store types of Prometheus queries for a given metric and grouping.