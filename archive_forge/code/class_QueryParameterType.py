from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryParameterType(_messages.Message):
    """The type of a query parameter.

  Messages:
    StructTypesValueListEntry: The type of a struct parameter.

  Fields:
    arrayType: Optional. The type of the array's elements, if this is an
      array.
    rangeElementType: Optional. The element type of the range, if this is a
      range.
    structTypes: Optional. The types of the fields of this struct, in order,
      if this is a struct.
    type: Required. The top level type of this field.
  """

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
    arrayType = _messages.MessageField('QueryParameterType', 1)
    rangeElementType = _messages.MessageField('QueryParameterType', 2)
    structTypes = _messages.MessageField('StructTypesValueListEntry', 3, repeated=True)
    type = _messages.StringField(4)