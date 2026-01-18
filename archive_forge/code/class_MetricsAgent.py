import json
import logging
import os
import re
import threading
import time
import traceback
from collections import namedtuple
from typing import List, Tuple, Any, Dict
from prometheus_client.core import (
from opencensus.metrics.export.value import ValueDouble
from opencensus.stats import aggregation
from opencensus.stats import measure as measure_module
from opencensus.stats.view_manager import ViewManager
from opencensus.stats.stats_recorder import StatsRecorder
from opencensus.stats.base_exporter import StatsExporter
from prometheus_client.core import Metric as PrometheusMetric
from opencensus.stats.aggregation_data import (
from opencensus.stats.view import View
from opencensus.tags import tag_key as tag_key_module
from opencensus.tags import tag_map as tag_map_module
from opencensus.tags import tag_value as tag_value_module
import ray
from ray._raylet import GcsClient
from ray.core.generated.metrics_pb2 import Metric
class MetricsAgent:

    def __init__(self, view_manager: ViewManager, stats_recorder: StatsRecorder, stats_exporter: StatsExporter=None):
        """A class to record and export metrics.

        The class exports metrics in 2 different ways.
        - Directly record and export metrics using OpenCensus.
        - Proxy metrics from other core components
            (e.g., raylet, GCS, core workers).

        This class is thread-safe.
        """
        self._lock = threading.Lock()
        self.view_manager = view_manager
        self.stats_recorder = stats_recorder
        self.stats_exporter = stats_exporter
        self.proxy_exporter_collector = None
        if self.stats_exporter is None:
            self.view_manager = None
        else:
            self.view_manager.register_exporter(stats_exporter)
            self.proxy_exporter_collector = OpenCensusProxyCollector(self.stats_exporter.options.namespace, component_timeout_s=int(os.getenv(RAY_WORKER_TIMEOUT_S, 120)))

    def record_and_export(self, records: List[Record], global_tags=None):
        """Directly record and export stats from the same process."""
        global_tags = global_tags or {}
        with self._lock:
            if not self.view_manager:
                return
            for record in records:
                gauge = record.gauge
                value = record.value
                tags = record.tags
                self._record_gauge(gauge, value, {**tags, **global_tags})

    def _record_gauge(self, gauge: Gauge, value: float, tags: dict):
        view_data = self.view_manager.get_view(gauge.name)
        if not view_data:
            self.view_manager.register_view(gauge.view)
        view = self.view_manager.get_view(gauge.name).view
        measurement_map = self.stats_recorder.new_measurement_map()
        tag_map = tag_map_module.TagMap()
        for key, tag_val in tags.items():
            tag_key = tag_key_module.TagKey(key)
            tag_value = tag_value_module.TagValue(tag_val)
            tag_map.insert(tag_key, tag_value)
        measurement_map.measure_float_put(view.measure, value)
        measurement_map.record(tag_map)

    def proxy_export_metrics(self, metrics: List[Metric], worker_id_hex: str=None):
        """Proxy export metrics specified by a Opencensus Protobuf.

        This API is used to export metrics emitted from
        core components.

        Args:
            metrics: A list of protobuf Metric defined from OpenCensus.
            worker_id_hex: The worker ID it proxies metrics export. None
                if the metric is not from a worker (i.e., raylet, GCS).
        """
        with self._lock:
            if not self.view_manager:
                return
        self._proxy_export_metrics(metrics, worker_id_hex)

    def _proxy_export_metrics(self, metrics: List[Metric], worker_id_hex: str=None):
        self.proxy_exporter_collector.record(metrics, worker_id_hex)

    def clean_all_dead_worker_metrics(self):
        """Clean dead worker's metrics.

        Worker metrics are cleaned up and won't be exported once
        it is considered as dead.

        This method has to be periodically called by a caller.
        """
        with self._lock:
            if not self.view_manager:
                return
        self.proxy_exporter_collector.clean_stale_components()