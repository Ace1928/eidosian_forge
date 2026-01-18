from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationCheckStatus(_messages.Message):
    """ValidationCheckStatus defines the detailed validation check status.

  Fields:
    result: Individual checks which failed as part of the Preflight check
      execution.
  """
    result = _messages.MessageField('ValidationCheckResult', 1, repeated=True)