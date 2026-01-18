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
class OpenMetricsMetric:
    """Container for all the COUNTER and GAUGE metrics extracted from an OpenMetrics endpoint."""

    def __init__(self, name: str, url: str, filters: Union[Mapping[str, Mapping[str, str]], Sequence[str], None]) -> None:
        self.name = name
        self.url = url
        self.filters = filters if isinstance(filters, Mapping) else {k: {} for k in filters or ['.*']}
        self.filters_tuple = _nested_dict_to_tuple(self.filters) if self.filters else ()
        self._session: Optional[requests.Session] = None
        self.samples: Deque[dict] = deque([])
        self.label_map: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.label_hashes: Dict[str, dict] = {}

    def setup(self) -> None:
        if self._session is not None:
            return
        self._session = _setup_requests_session()

    def teardown(self) -> None:
        if self._session is None:
            return
        self._session.close()
        self._session = None

    def parse_open_metrics_endpoint(self) -> Dict[str, Union[str, int, float]]:
        assert prometheus_client_parser is not None
        assert self._session is not None
        response = self._session.get(self.url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        text = response.text
        measurement = {}
        for family in prometheus_client_parser.text_string_to_metric_families(text):
            if family.type not in ('counter', 'gauge'):
                continue
            for sample in family.samples:
                name, labels, value = (sample.name, sample.labels, sample.value)
                if not _should_capture_metric(self.name, name, tuple(labels.items()), self.filters_tuple):
                    continue
                label_hash = hashutil._md5(str(labels).encode('utf-8')).hexdigest()
                if label_hash not in self.label_map[name]:
                    self.label_map[name][label_hash] = len(self.label_map[name])
                    self.label_hashes[label_hash] = labels
                index = self.label_map[name][label_hash]
                measurement[f'{name}.{index}'] = value
        return measurement

    def sample(self) -> None:
        s = self.parse_open_metrics_endpoint()
        self.samples.append(s)

    def clear(self) -> None:
        self.samples.clear()

    def aggregate(self) -> dict:
        if not self.samples:
            return {}
        prefix = f'{_PREFIX}.{self.name}.'
        stats = {}
        for key in self.samples[0].keys():
            samples = [s[key] for s in self.samples if key in s]
            if samples and all((isinstance(s, (int, float)) for s in samples)):
                stats[f'{prefix}{key}'] = aggregate_mean(samples)
            else:
                stats[f'{prefix}{key}'] = aggregate_last(samples)
        return stats