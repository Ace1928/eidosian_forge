import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
class UnknownMetricFamily(Metric):
    """A single unknown metric and its samples.
    For use by custom collectors.
    """

    def __init__(self, name: str, documentation: str, value: Optional[float]=None, labels: Optional[Sequence[str]]=None, unit: str=''):
        Metric.__init__(self, name, documentation, 'unknown', unit)
        if labels is not None and value is not None:
            raise ValueError('Can only specify at most one of value and labels.')
        if labels is None:
            labels = []
        self._labelnames = tuple(labels)
        if value is not None:
            self.add_metric([], value)

    def add_metric(self, labels: Sequence[str], value: float, timestamp: Optional[Union[Timestamp, float]]=None) -> None:
        """Add a metric to the metric family.
        Args:
        labels: A list of label values
        value: The value of the metric.
        """
        self.samples.append(Sample(self.name, dict(zip(self._labelnames, labels)), value, timestamp))