from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActivationFailed(_messages.Message):
    """An event representing that the Grant activation failed.

  Fields:
    error: Output only. The error that occurred while activating the Grant.
  """
    error = _messages.MessageField('Status', 1)