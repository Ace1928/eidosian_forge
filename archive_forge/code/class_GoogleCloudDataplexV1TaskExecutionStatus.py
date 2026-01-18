from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskExecutionStatus(_messages.Message):
    """Status of the task execution (e.g. Jobs).

  Fields:
    latestJob: Output only. latest job execution
    updateTime: Output only. Last update time of the status.
  """
    latestJob = _messages.MessageField('GoogleCloudDataplexV1Job', 1)
    updateTime = _messages.StringField(2)