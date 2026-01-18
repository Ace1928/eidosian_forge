import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class SummaryMetricFamily(Metric):
    """A single summary and its samples.

    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, count_value: Optional[int]=None, sum_value: Optional[float]=None, labels: Optional[Sequence[str]]=None, unit: str=''):
        Metric.__init__(self, name, documentation, 'summary', unit)
        if (sum_value is None) != (count_value is None):
            raise ValueError('count_value and sum_value must be provided together.')
        if labels is not None and count_value is not None:
            raise ValueError('Can only specify at most one of value and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if count_value is not None and sum_value is not None:
            self.add_metric([], count_value, sum_value)

    def add_metric(self, labels: Sequence[str], count_value: int, sum_value: float, timestamp: Optional[Union[float, Timestamp]]=None) -> None:
        """Add a metric to the metric family.

        Args:
          labels: A list of label values
          count_value: The count value of the metric.
          sum_value: The sum value of the metric.
        """
        self.samples.append(Sample(self.name + '_count', dict(zip(self._labelnames, labels)), count_value, timestamp))
        self.samples.append(Sample(self.name + '_sum', dict(zip(self._labelnames, labels)), sum_value, timestamp))