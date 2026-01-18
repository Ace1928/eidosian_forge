from __future__ import annotations
from typing import TYPE_CHECKING
import tornado.web
from streamlit.runtime.stats import CacheStat, StatsManager
from streamlit.web.server.server_util import emit_endpoint_deprecation_notice
@staticmethod
def _stats_to_proto(stats: list[CacheStat]) -> MetricSetProto:
    from streamlit.proto.openmetrics_data_model_pb2 import GAUGE
    from streamlit.proto.openmetrics_data_model_pb2 import MetricSet as MetricSetProto
    metric_set = MetricSetProto()
    metric_family = metric_set.metric_families.add()
    metric_family.name = 'cache_memory_bytes'
    metric_family.type = GAUGE
    metric_family.unit = 'bytes'
    metric_family.help = 'Total memory consumed by a cache.'
    for stat in stats:
        metric_proto = metric_family.metrics.add()
        stat.marshall_metric_proto(metric_proto)
    metric_set = MetricSetProto()
    metric_set.metric_families.append(metric_family)
    return metric_set