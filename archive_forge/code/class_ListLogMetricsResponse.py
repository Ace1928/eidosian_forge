from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListLogMetricsResponse(_messages.Message):
    """Result returned from ListLogMetrics.

  Fields:
    metrics: A list of logs-based metrics.
    nextPageToken: If there might be more results than appear in this
      response, then nextPageToken is included. To get the next set of
      results, call this method again using the value of nextPageToken as
      pageToken.
  """
    metrics = _messages.MessageField('LogMetric', 1, repeated=True)
    nextPageToken = _messages.StringField(2)