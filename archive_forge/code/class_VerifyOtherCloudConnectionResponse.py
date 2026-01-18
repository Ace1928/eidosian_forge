from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VerifyOtherCloudConnectionResponse(_messages.Message):
    """Response to verify an other-cloud connection.

  Fields:
    validationResult: The validation result of the other-cloud connection.
  """
    validationResult = _messages.MessageField('ValidationResult', 1)