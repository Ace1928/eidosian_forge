from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1QueryMetadata(_messages.Message):
    """A GoogleCloudApigeeV1QueryMetadata object.

  Fields:
    dimensions: Dimensions of the AsyncQuery.
    endTimestamp: End timestamp of the query range.
    metrics: Metrics of the AsyncQuery. Example:
      ["name:message_count,func:sum,alias:sum_message_count"]
    outputFormat: Output format.
    startTimestamp: Start timestamp of the query range.
    timeUnit: Query GroupBy time unit.
  """
    dimensions = _messages.StringField(1, repeated=True)
    endTimestamp = _messages.StringField(2)
    metrics = _messages.StringField(3, repeated=True)
    outputFormat = _messages.StringField(4)
    startTimestamp = _messages.StringField(5)
    timeUnit = _messages.StringField(6)