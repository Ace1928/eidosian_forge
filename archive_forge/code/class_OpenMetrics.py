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
class OpenMetrics:

    def __init__(self, interface: 'Interface', settings: 'SettingsStatic', shutdown_event: threading.Event, name: str, url: str) -> None:
        self.name = name
        self.url = url
        self.interface = interface
        self.settings = settings
        self.shutdown_event = shutdown_event
        self.metrics: List[Metric] = [OpenMetricsMetric(name, url, settings._stats_open_metrics_filters)]
        self.metrics_monitor: MetricsMonitor = MetricsMonitor(asset_name=self.name, metrics=self.metrics, interface=interface, settings=settings, shutdown_event=shutdown_event)
        telemetry_record = telemetry.TelemetryRecord()
        telemetry_record.feature.open_metrics = True
        interface._publish_telemetry(telemetry_record)

    @classmethod
    def is_available(cls, url: str) -> bool:
        _is_available: bool = False
        ret = prometheus_client_parser is not None
        if not ret:
            wandb.termwarn('Monitoring OpenMetrics endpoints requires the `prometheus_client` package. To install it, run `pip install prometheus_client`.', repeat=False)
            return _is_available
        _session: Optional[requests.Session] = None
        try:
            assert prometheus_client_parser is not None
            _session = _setup_requests_session()
            response = _session.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
            if list(prometheus_client_parser.text_string_to_metric_families(response.text)):
                _is_available = True
        except Exception as e:
            logger.debug(f'OpenMetrics endpoint {url} is not available: {e}', exc_info=True)
        if _session is not None:
            try:
                _session.close()
            except Exception:
                pass
        return _is_available

    def start(self) -> None:
        self.metrics_monitor.start()

    def finish(self) -> None:
        self.metrics_monitor.finish()

    def probe(self) -> dict:
        return {self.name: self.url}