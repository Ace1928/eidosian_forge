import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class HistogramMetricFamily(Metric):
    """A single histogram and its samples.

    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, buckets: Optional[Sequence[Union[Tuple[str, float], Tuple[str, float, Exemplar]]]]=None, sum_value: Optional[float]=None, labels: Optional[Sequence[str]]=None, unit: str=''):
        Metric.__init__(self, name, documentation, 'histogram', unit)
        if sum_value is not None and buckets is None:
            raise ValueError('sum value cannot be provided without buckets.')
        if labels is not None and buckets is not None:
            raise ValueError('Can only specify at most one of buckets and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if buckets is not None:
            self.add_metric([], buckets, sum_value)

    def add_metric(self, labels: Sequence[str], buckets: Sequence[Union[Tuple[str, float], Tuple[str, float, Exemplar]]], sum_value: Optional[float], timestamp: Optional[Union[Timestamp, float]]=None) -> None:
        """Add a metric to the metric family.

        Args:
          labels: A list of label values
          buckets: A list of lists.
              Each inner list can be a pair of bucket name and value,
              or a triple of bucket name, value, and exemplar.
              The buckets must be sorted, and +Inf present.
          sum_value: The sum value of the metric.
        """
        for b in buckets:
            bucket, value = b[:2]
            exemplar = None
            if len(b) == 3:
                exemplar = b[2]
            self.samples.append(Sample(self.name + '_bucket', dict(list(zip(self._labelnames, labels)) + [('le', bucket)]), value, timestamp, exemplar))
        if float(buckets[0][0]) >= 0 and sum_value is not None:
            self.samples.append(Sample(self.name + '_count', dict(zip(self._labelnames, labels)), buckets[-1][1], timestamp))
            self.samples.append(Sample(self.name + '_sum', dict(zip(self._labelnames, labels)), sum_value, timestamp))