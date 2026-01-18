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
class OpencensusProxyMetric:

    def __init__(self, name: str, desc: str, unit: str, label_keys: List[str]):
        """Represents the OpenCensus metrics that will be proxy exported."""
        self._name = name
        self._desc = desc
        self._unit = unit
        self._label_keys = label_keys
        self._data = {}

    @property
    def name(self):
        return self._name

    @property
    def desc(self):
        return self._desc

    @property
    def unit(self):
        return self._unit

    @property
    def label_keys(self):
        return self._label_keys

    @property
    def data(self):
        return self._data

    def record(self, metric: Metric):
        """Parse the Opencensus Protobuf and store the data.

        The data can be accessed via `data` API once recorded.
        """
        timeseries = metric.timeseries
        if len(timeseries) == 0:
            return
        for series in timeseries:
            labels = tuple((val.value for val in series.label_values))
            for point in series.points:
                if point.HasField('int64_value'):
                    data = CountAggregationData(point.int64_value)
                elif point.HasField('double_value'):
                    data = LastValueAggregationData(ValueDouble, point.double_value)
                elif point.HasField('distribution_value'):
                    dist_value = point.distribution_value
                    counts_per_bucket = [bucket.count for bucket in dist_value.buckets]
                    bucket_bounds = dist_value.bucket_options.explicit.bounds
                    data = DistributionAggregationData(dist_value.sum / dist_value.count, dist_value.count, dist_value.sum_of_squared_deviation, counts_per_bucket, bucket_bounds)
                else:
                    raise ValueError('Summary is not supported')
                self._data[labels] = data