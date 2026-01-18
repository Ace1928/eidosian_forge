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
class OpenCensusProxyCollector:

    def __init__(self, namespace: str, component_timeout_s: int=60):
        """Prometheus collector implementation for opencensus proxy export.

        Prometheus collector requires to implement `collect` which is
        invoked whenever Prometheus queries the endpoint.

        The class is thread-safe.

        Args:
            namespace: Prometheus namespace.
        """
        self._components_lock = threading.Lock()
        self._component_timeout_s = component_timeout_s
        self._namespace = namespace
        self._components = {}

    def record(self, metrics: List[Metric], worker_id_hex: str=None):
        """Record the metrics reported from the component that reports it.

        Args:
            metrics: A list of opencensus protobuf to proxy export metrics.
            worker_id_hex: A worker id that reports these metrics.
                If None, it means they are reported from Raylet or GCS.
        """
        key = GLOBAL_COMPONENT_KEY if not worker_id_hex else worker_id_hex
        with self._components_lock:
            if key not in self._components:
                self._components[key] = Component(key)
            self._components[key].record(metrics)

    def clean_stale_components(self):
        """Clean up stale components.

        Stale means the component is dead or unresponsive.

        Stale components won't be reported to Prometheus anymore.
        """
        with self._components_lock:
            stale_components = []
            stale_component_ids = []
            for id, component in self._components.items():
                elapsed = time.monotonic() - component.last_reported_time
                if elapsed > self._component_timeout_s:
                    stale_component_ids.append(id)
                    logger.info('Metrics from a worker ({}) is cleaned up due to timeout. Time since last report {}s'.format(id, elapsed))
            for id in stale_component_ids:
                stale_components.append(self._components.pop(id))
            return stale_components

    def to_metric(self, metric_name: str, metric_description: str, label_keys: List[str], metric_units: str, label_values: Tuple[tag_value_module.TagValue], agg_data: Any, metrics_map: Dict[str, PrometheusMetric]) -> PrometheusMetric:
        """to_metric translate the data that OpenCensus create
        to Prometheus format, using Prometheus Metric object.

        This method is from Opencensus Prometheus Exporter.

        Args:
            metric_name: Name of the metric.
            metric_description: Description of the metric.
            label_keys: The fixed label keys of the metric.
            metric_units: Units of the metric.
            label_values: The values of `label_keys`.
            agg_data: `opencensus.stats.aggregation_data.AggregationData` object.
                Aggregated data that needs to be converted as Prometheus samples

        Returns:
            A Prometheus metric object
        """
        assert self._components_lock.locked()
        metric_name = f'{self._namespace}_{metric_name}'
        assert len(label_values) == len(label_keys), (label_values, label_keys)
        label_values = [tv if tv else '' for tv in label_values]
        if isinstance(agg_data, CountAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = CounterMetricFamily(name=metric_name, documentation=metric_description, unit=metric_units, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=label_values, value=agg_data.count_data)
            return metric
        elif isinstance(agg_data, DistributionAggregationData):
            assert agg_data.bounds == sorted(agg_data.bounds)
            buckets = []
            cum_count = 0
            for ii, bound in enumerate(agg_data.bounds):
                cum_count += agg_data.counts_per_bucket[ii]
                bucket = [str(bound), cum_count]
                buckets.append(bucket)
            buckets.append(['+Inf', agg_data.count_data])
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = HistogramMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=label_values, buckets=buckets, sum_value=agg_data.sum)
            return metric
        elif isinstance(agg_data, LastValueAggregationData):
            metric = metrics_map.get(metric_name)
            if not metric:
                metric = GaugeMetricFamily(name=metric_name, documentation=metric_description, labels=label_keys)
                metrics_map[metric_name] = metric
            metric.add_metric(labels=label_values, value=agg_data.value)
            return metric
        else:
            raise ValueError(f'unsupported aggregation type {type(agg_data)}')

    def collect(self):
        """Collect fetches the statistics from OpenCensus
        and delivers them as Prometheus Metrics.
        Collect is invoked every time a prometheus.Gatherer is run
        for example when the HTTP endpoint is invoked by Prometheus.

        This method is required as a Prometheus Collector.
        """
        with self._components_lock:
            metrics_map = {}
            for component in self._components.values():
                for metric in component.metrics.values():
                    for label_values, data in metric.data.items():
                        self.to_metric(metric.name, metric.desc, metric.label_keys, metric.unit, label_values, data, metrics_map)
        for metric in metrics_map.values():
            yield metric