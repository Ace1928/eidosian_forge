from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeekSubscriptionRequest(_messages.Message):
    """Request for SeekSubscription.

  Enums:
    NamedTargetValueValuesEnum: Seek to a named position with respect to the
      message backlog.

  Fields:
    namedTarget: Seek to a named position with respect to the message backlog.
    timeTarget: Seek to the first message whose publish or event time is
      greater than or equal to the specified query time. If no such message
      can be located, will seek to the end of the message backlog.
  """

    class NamedTargetValueValuesEnum(_messages.Enum):
        """Seek to a named position with respect to the message backlog.

    Values:
      NAMED_TARGET_UNSPECIFIED: Unspecified named target. Do not use.
      TAIL: Seek to the oldest retained message.
      HEAD: Seek past all recently published messages, skipping the entire
        message backlog.
    """
        NAMED_TARGET_UNSPECIFIED = 0
        TAIL = 1
        HEAD = 2
    namedTarget = _messages.EnumField('NamedTargetValueValuesEnum', 1)
    timeTarget = _messages.MessageField('TimeTarget', 2)