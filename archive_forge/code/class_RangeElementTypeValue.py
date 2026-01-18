from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RangeElementTypeValue(_messages.Message):
    """Represents the type of a field element.

    Fields:
      type: Required. The type of a field element. For more information, see
        TableFieldSchema.type.
    """
    type = _messages.StringField(1)