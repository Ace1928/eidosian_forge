from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StagePerformanceChangeInsight(_messages.Message):
    """Performance insights compared to the previous executions for a specific
  stage.

  Fields:
    inputDataChange: Output only. Input data change insight of the query
      stage.
    stageId: Output only. The stage id that the insight mapped to.
  """
    inputDataChange = _messages.MessageField('InputDataChange', 1)
    stageId = _messages.IntegerField(2)