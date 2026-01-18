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
@lru_cache(maxsize=128)
def _should_capture_metric(endpoint_name: str, metric_name: str, metric_labels: Tuple[str, ...], filters: Tuple[Tuple[str, Tuple[str, str]], ...]) -> bool:
    should_capture = False
    if not filters:
        return should_capture
    metric_labels_dict = {t[0]: t[1] for t in metric_labels}
    filters_dict = _tuple_to_nested_dict(filters)
    for metric_name_regex, label_filters in filters_dict.items():
        if not re.match(metric_name_regex, f'{endpoint_name}.{metric_name}'):
            continue
        should_capture = True
        for label, label_filter in label_filters.items():
            if not re.match(label_filter, metric_labels_dict.get(label, '')):
                should_capture = False
                break
        break
    return should_capture