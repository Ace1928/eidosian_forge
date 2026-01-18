from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ANR(_messages.Message):
    """Additional details for an ANR crash.

  Fields:
    stackTrace: The stack trace of the ANR crash. Optional.
  """
    stackTrace = _messages.MessageField('StackTrace', 1)