from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerfSampleSeries(_messages.Message):
    """Resource representing a collection of performance samples (or data
  points)

  Fields:
    basicPerfSampleSeries: Basic series represented by a line chart
    executionId: A tool results execution ID. @OutputOnly
    historyId: A tool results history ID. @OutputOnly
    projectId: The cloud project @OutputOnly
    sampleSeriesId: A sample series id @OutputOnly
    stepId: A tool results step ID. @OutputOnly
  """
    basicPerfSampleSeries = _messages.MessageField('BasicPerfSampleSeries', 1)
    executionId = _messages.StringField(2)
    historyId = _messages.StringField(3)
    projectId = _messages.StringField(4)
    sampleSeriesId = _messages.StringField(5)
    stepId = _messages.StringField(6)