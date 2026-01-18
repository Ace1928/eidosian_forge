import logging
import re
import sys
import threading
from collections import defaultdict, deque
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Tuple, Union
import requests
import requests.adapters
import urllib3
import wandb
from wandb.sdk.lib import hashutil, telemetry
from .aggregators import aggregate_last, aggregate_mean
from .interfaces import Interface, Metric, MetricsMonitor
def _setup_requests_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=_REQUEST_RETRY_STRATEGY, pool_connections=_REQUEST_POOL_CONNECTIONS, pool_maxsize=_REQUEST_POOL_MAXSIZE)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session