from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class RowsValueListEntry(_messages.Message):
    """Single entry in a RowsValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
    entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)