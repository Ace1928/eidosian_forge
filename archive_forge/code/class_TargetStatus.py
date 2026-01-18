from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetStatus(_messages.Message):
    """Message describing target status.

  Fields:
    status: Output only. Error message from the control plane.
    target: target that apphub target resolved to.
  """
    status = _messages.MessageField('Status', 1)
    target = _messages.MessageField('Target', 2)