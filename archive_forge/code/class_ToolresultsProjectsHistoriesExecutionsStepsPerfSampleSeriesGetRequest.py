from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesGetRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesGetRequest
  object.

  Fields:
    executionId: A tool results execution ID.
    historyId: A tool results history ID.
    projectId: The cloud project
    sampleSeriesId: A sample series id
    stepId: A tool results step ID.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    sampleSeriesId = _messages.StringField(4, required=True)
    stepId = _messages.StringField(5, required=True)