from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StructTypesValueListEntry(_messages.Message):
    """The type of a struct parameter.

    Fields:
      description: Optional. Human-oriented description of the field.
      name: Optional. The name of this field.
      type: Required. The type of this field.
    """
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    type = _messages.MessageField('QueryParameterType', 3)