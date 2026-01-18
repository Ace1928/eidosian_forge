from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailedData(_messages.Message):
    """Further data for the failed state.

  Fields:
    error: Output only. The error that caused the queued resource to enter the
      FAILED state.
  """
    error = _messages.MessageField('Status', 1)