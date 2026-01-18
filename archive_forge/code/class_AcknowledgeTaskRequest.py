from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AcknowledgeTaskRequest(_messages.Message):
    """Request message for acknowledging a task using AcknowledgeTask.

  Fields:
    scheduleTime: Required. The task's current schedule time, available in the
      schedule_time returned by LeaseTasks response or RenewLease response.
      This restriction is to ensure that your worker currently holds the
      lease.
  """
    scheduleTime = _messages.StringField(1)