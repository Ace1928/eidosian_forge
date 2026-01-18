from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplaceAccessLevelsResponse(_messages.Message):
    """A response to ReplaceAccessLevelsRequest. This will be put inside of
  Operation.response field.

  Fields:
    accessLevels: List of the Access Level instances.
  """
    accessLevels = _messages.MessageField('AccessLevel', 1, repeated=True)