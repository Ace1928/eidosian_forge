from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepairMode(_messages.Message):
    """Configuration of the repair action.

  Fields:
    retry: Optional. Retries a failed job.
    rollback: Optional. Rolls back a `Rollout`.
  """
    retry = _messages.MessageField('Retry', 1)
    rollback = _messages.MessageField('Rollback', 2)