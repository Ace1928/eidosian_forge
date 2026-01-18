from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttemptStatus(_messages.Message):
    """The status of a task attempt.

  Fields:
    dispatchTime: Output only. The time that this attempt was dispatched.
      `dispatch_time` will be truncated to the nearest microsecond.
    responseStatus: Output only. The response from the target for this
      attempt. If the task has not been attempted or the task is currently
      running then the response status is unset.
    responseTime: Output only. The time that this attempt response was
      received. `response_time` will be truncated to the nearest microsecond.
    scheduleTime: Output only. The time that this attempt was scheduled.
      `schedule_time` will be truncated to the nearest microsecond.
  """
    dispatchTime = _messages.StringField(1)
    responseStatus = _messages.MessageField('Status', 2)
    responseTime = _messages.StringField(3)
    scheduleTime = _messages.StringField(4)