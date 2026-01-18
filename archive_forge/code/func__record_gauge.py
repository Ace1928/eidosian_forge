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