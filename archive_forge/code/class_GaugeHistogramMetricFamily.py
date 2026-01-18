import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class GaugeHistogramMetricFamily(Metric):
    """A single gauge histogram and its samples.

    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, buckets: Optional[Sequence[Tuple[str, float]]]=None, gsum_value: Optional[float]=None, labels: Optional[Sequence[str]]=None, unit: str=''):
        Metric.__init__(self, name, documentation, 'gaugehistogram', unit)
        if labels is not None and buckets is not None:
            raise ValueError('Can only specify at most one of buckets and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if buckets is not None:
            self.add_metric([], buckets, gsum_value)

    def add_metric(self, labels: Sequence[str], buckets: Sequence[Tuple[str, float]], gsum_value: Optional[float], timestamp: Optional[Union[float, Timestamp]]=None) -> None:
        """Add a metric to the metric family.

        Args:
          labels: A list of label values
          buckets: A list of pairs of bucket names and values.
              The buckets must be sorted, and +Inf present.
          gsum_value: The sum value of the metric.
        """
        for bucket, value in buckets:
            self.samples.append(Sample(self.name + '_bucket', dict(list(zip(self._labelnames, labels)) + [('le', bucket)]), value, timestamp))
        self.samples.extend([Sample(self.name + '_gcount', dict(zip(self._labelnames, labels)), buckets[-1][1], timestamp), Sample(self.name + '_gsum', dict(zip(self._labelnames, labels)), gsum_value, timestamp)])