from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryParameter(_messages.Message):
    """A parameter given to a query.

  Fields:
    name: Optional. If unset, this is a positional parameter. Otherwise,
      should be unique within a query.
    parameterType: Required. The type of this parameter.
    parameterValue: Required. The value of this parameter.
  """
    name = _messages.StringField(1)
    parameterType = _messages.MessageField('QueryParameterType', 2)
    parameterValue = _messages.MessageField('QueryParameterValue', 3)