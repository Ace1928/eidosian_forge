from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActiveTimeRangesValueListEntry(_messages.Message):
    """A ActiveTimeRangesValueListEntry object.

    Fields:
      activeTime: Duration in milliseconds
      date: Date of usage
    """
    activeTime = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    date = extra_types.DateField(2)