from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprMapValueEntry(_messages.Message):
    """A GoogleApiExprMapValueEntry object.

  Fields:
    key: The key. Must be unique with in the map. Currently only boolean, int,
      uint, and string values can be keys.
    value: The value.
  """
    key = _messages.MessageField('GoogleApiExprValue', 1)
    value = _messages.MessageField('GoogleApiExprValue', 2)