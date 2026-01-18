from __future__ import annotations
import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable
class CacheStat(NamedTuple):
    """Describes a single cache entry.

    Properties
    ----------
    category_name : str
        A human-readable name for the cache "category" that the entry belongs
        to - e.g. "st.memo", "session_state", etc.
    cache_name : str
        A human-readable name for cache instance that the entry belongs to.
        For "st.memo" and other function decorator caches, this might be the
        name of the cached function. If the cache category doesn't have
        multiple separate cache instances, this can just be the empty string.
    byte_length : int
        The entry's memory footprint in bytes.
    """
    category_name: str
    cache_name: str
    byte_length: int

    def to_metric_str(self) -> str:
        return f'cache_memory_bytes{{cache_type="{self.category_name}",cache="{self.cache_name}"}} {self.byte_length}'

    def marshall_metric_proto(self, metric: MetricProto) -> None:
        """Fill an OpenMetrics `Metric` protobuf object."""
        label = metric.labels.add()
        label.name = 'cache_type'
        label.value = self.category_name
        label = metric.labels.add()
        label.name = 'cache'
        label.value = self.cache_name
        metric_point = metric.metric_points.add()
        metric_point.gauge_value.int_value = self.byte_length