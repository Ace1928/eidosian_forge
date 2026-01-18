from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateCreateMembershipResponse(_messages.Message):
    """Response message for the `GkeHub.ValidateCreateMembership` method.

  Fields:
    validationResults: Wraps all the validator results.
  """
    validationResults = _messages.MessageField('ValidationResult', 1, repeated=True)