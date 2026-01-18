import re
from typing import Dict, List, Optional, Sequence, Tuple, Union
from .samples import Exemplar, Sample, Timestamp
def _restricted_metric(self, names):
    """Build a snapshot of a metric with samples restricted to a given set of names."""
    samples = [s for s in self.samples if s[0] in names]
    if samples:
        m = Metric(self.name, self.documentation, self.type)
        m.samples = samples
        return m
    return None