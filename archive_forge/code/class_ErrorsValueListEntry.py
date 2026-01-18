from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorsValueListEntry(_messages.Message):
    """A ErrorsValueListEntry object.

      Fields:
        code: [Output Only] The error type identifier for this error.
        location: [Output Only] Indicates the field in the request that caused
          the error. This property is optional.
        message: [Output Only] An optional, human-readable error message.
      """
    code = _messages.StringField(1)
    location = _messages.StringField(2)
    message = _messages.StringField(3)