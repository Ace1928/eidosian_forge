from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricKindValueValuesEnum(_messages.Enum):
    """Whether the metric records instantaneous values, changes to a value,
    etc.

    Values:
      METRIC_KIND_UNSPECIFIED: Do not use this default value.
      GAUGE: Instantaneous measurements of a varying quantity.
      DELTA: Changes over non-overlapping time intervals.
      CUMULATIVE: Cumulative value over time intervals that can overlap. The
        overlapping intervals must have the same start time.
    """
    METRIC_KIND_UNSPECIFIED = 0
    GAUGE = 1
    DELTA = 2
    CUMULATIVE = 3