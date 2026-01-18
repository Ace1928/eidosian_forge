from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingValue(_messages.Message):
    """The bucket's logging configuration, which defines the destination
    bucket and optional name prefix for the current bucket's logs.

    Fields:
      logBucket: The destination bucket where the current bucket's logs should
        be placed.
      logObjectPrefix: A prefix for log object names.
    """
    logBucket = _messages.StringField(1)
    logObjectPrefix = _messages.StringField(2)