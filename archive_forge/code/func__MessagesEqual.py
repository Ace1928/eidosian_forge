import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def _MessagesEqual(msg1, msg2):
    """Compare two protorpc messages for equality.

    Using python's == operator does not work in all cases, specifically when
    there is a list involved.

    Args:
      msg1: protorpc.messages.Message or [protorpc.messages.Message] or number
          or string, One of the messages to compare.
      msg2: protorpc.messages.Message or [protorpc.messages.Message] or number
          or string, One of the messages to compare.

    Returns:
      If the messages are isomorphic.
    """
    if isinstance(msg1, list) and isinstance(msg2, list):
        if len(msg1) != len(msg2):
            return False
        return all((_MessagesEqual(x, y) for x, y in zip(msg1, msg2)))
    if not isinstance(msg1, messages.Message) or not isinstance(msg2, messages.Message):
        return msg1 == msg2
    for field in msg1.all_fields():
        field1 = getattr(msg1, field.name)
        field2 = getattr(msg2, field.name)
        if not _MessagesEqual(field1, field2):
            return False
    return True