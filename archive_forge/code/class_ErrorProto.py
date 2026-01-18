from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorProto(_messages.Message):
    """A ErrorProto object.

  Fields:
    debugInfo: Debugging information. This property is internal to Google and
      should not be used.
    location: Specifies where the error occurred, if present.
    message: A human-readable description of the error.
    reason: A short error code that summarizes the error.
  """
    debugInfo = _messages.StringField(1)
    location = _messages.StringField(2)
    message = _messages.StringField(3)
    reason = _messages.StringField(4)