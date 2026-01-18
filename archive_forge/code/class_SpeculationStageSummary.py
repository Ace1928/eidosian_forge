from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeculationStageSummary(_messages.Message):
    """Details of the speculation task when speculative execution is enabled.

  Fields:
    numActiveTasks: A integer attribute.
    numCompletedTasks: A integer attribute.
    numFailedTasks: A integer attribute.
    numKilledTasks: A integer attribute.
    numTasks: A integer attribute.
    stageAttemptId: A integer attribute.
    stageId: A string attribute.
  """
    numActiveTasks = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    numCompletedTasks = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    numFailedTasks = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    numKilledTasks = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    numTasks = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    stageAttemptId = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(7)