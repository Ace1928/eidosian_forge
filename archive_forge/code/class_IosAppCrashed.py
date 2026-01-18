from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IosAppCrashed(_messages.Message):
    """Additional details for an iOS app crash.

  Fields:
    stackTrace: The stack trace, if one is available. Optional.
  """
    stackTrace = _messages.MessageField('StackTrace', 1)