from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueuedResourceStatusFailedData(_messages.Message):
    """Additional status detail for the FAILED state.

  Messages:
    ErrorValue: The error(s) that caused the QueuedResource to enter the
      FAILED state.

  Fields:
    error: The error(s) that caused the QueuedResource to enter the FAILED
      state.
  """

    class ErrorValue(_messages.Message):
        """The error(s) that caused the QueuedResource to enter the FAILED state.

    Messages:
      ErrorsValueListEntry: A ErrorsValueListEntry object.

    Fields:
      errors: [Output Only] The array of errors encountered while processing
        this operation.
    """

        class ErrorsValueListEntry(_messages.Message):
            """A ErrorsValueListEntry object.

      Messages:
        ErrorDetailsValueListEntry: A ErrorDetailsValueListEntry object.

      Fields:
        code: [Output Only] The error type identifier for this error.
        errorDetails: [Output Only] An optional list of messages that contain
          the error details. There is a set of defined message types to use
          for providing details.The syntax depends on the error code. For
          example, QuotaExceededInfo will have details when the error code is
          QUOTA_EXCEEDED.
        location: [Output Only] Indicates the field in the request that caused
          the error. This property is optional.
        message: [Output Only] An optional, human-readable error message.
      """

            class ErrorDetailsValueListEntry(_messages.Message):
                """A ErrorDetailsValueListEntry object.

        Fields:
          errorInfo: A ErrorInfo attribute.
          help: A Help attribute.
          localizedMessage: A LocalizedMessage attribute.
          quotaInfo: A QuotaExceededInfo attribute.
        """
                errorInfo = _messages.MessageField('ErrorInfo', 1)
                help = _messages.MessageField('Help', 2)
                localizedMessage = _messages.MessageField('LocalizedMessage', 3)
                quotaInfo = _messages.MessageField('QuotaExceededInfo', 4)
            code = _messages.StringField(1)
            errorDetails = _messages.MessageField('ErrorDetailsValueListEntry', 2, repeated=True)
            location = _messages.StringField(3)
            message = _messages.StringField(4)
        errors = _messages.MessageField('ErrorsValueListEntry', 1, repeated=True)
    error = _messages.MessageField('ErrorValue', 1)