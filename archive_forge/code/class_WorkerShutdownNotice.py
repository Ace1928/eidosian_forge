from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerShutdownNotice(_messages.Message):
    """Shutdown notification from workers. This is to be sent by the shutdown
  script of the worker VM so that the backend knows that the VM is being shut
  down.

  Fields:
    reason: The reason for the worker shutdown. Current possible values are:
      "UNKNOWN": shutdown reason is unknown. "PREEMPTION": shutdown reason is
      preemption. Other possible reasons may be added in the future.
  """
    reason = _messages.StringField(1)