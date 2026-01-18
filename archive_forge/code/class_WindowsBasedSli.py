from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowsBasedSli(_messages.Message):
    """A WindowsBasedSli defines good_service as the count of time windows for
  which the provided service was of good quality. Criteria for determining if
  service was good are embedded in the window_criterion.

  Fields:
    goodBadMetricFilter: A monitoring filter
      (https://cloud.google.com/monitoring/api/v3/filters) specifying a
      TimeSeries with ValueType = BOOL. The window is good if any true values
      appear in the window.
    goodTotalRatioThreshold: A window is good if its performance is high
      enough.
    metricMeanInRange: A window is good if the metric's value is in a good
      range, averaged across returned streams.
    metricSumInRange: A window is good if the metric's value is in a good
      range, summed across returned streams.
    windowPeriod: Duration over which window quality is evaluated. Must be an
      integer fraction of a day and at least 60s.
  """
    goodBadMetricFilter = _messages.StringField(1)
    goodTotalRatioThreshold = _messages.MessageField('PerformanceThreshold', 2)
    metricMeanInRange = _messages.MessageField('MetricRange', 3)
    metricSumInRange = _messages.MessageField('MetricRange', 4)
    windowPeriod = _messages.StringField(5)