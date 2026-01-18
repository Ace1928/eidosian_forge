from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorValue(_messages.Message):
    """Output only. If errors are generated during update of the resource,
    this field will be populated.

    Messages:
      ErrorsValueListEntry: A ErrorsValueListEntry object.

    Fields:
      errors: [Output Only] The array of errors encountered while processing
        this operation.
    """

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
    errors = _messages.MessageField('ErrorsValueListEntry', 1, repeated=True)