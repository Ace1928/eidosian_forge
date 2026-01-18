from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineReadyCondition(_messages.Message):
    """PipelineReadyCondition contains information around the status of the
  Pipeline.

  Fields:
    status: True if the Pipeline is in a valid state. Otherwise at least one
      condition in `PipelineCondition` is in an invalid state. Iterate over
      those conditions and see which condition(s) has status = false to find
      out what is wrong with the Pipeline.
    updateTime: Last time the condition was updated.
  """
    status = _messages.BooleanField(1)
    updateTime = _messages.StringField(2)