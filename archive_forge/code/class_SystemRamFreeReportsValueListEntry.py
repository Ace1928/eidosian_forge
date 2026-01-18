from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SystemRamFreeReportsValueListEntry(_messages.Message):
    """A SystemRamFreeReportsValueListEntry object.

    Fields:
      reportTime: Date and time the report was received.
      systemRamFreeInfo: A string attribute.
    """
    reportTime = _message_types.DateTimeField(1)
    systemRamFreeInfo = _messages.IntegerField(2, repeated=True)