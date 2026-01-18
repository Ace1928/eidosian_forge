from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprCreateStruct(_messages.Message):
    """A map or message creation expression. Maps are constructed as
  `{'key_name': 'value'}`. Message construction is similar, but prefixed with
  a type name and composed of field ids: `types.MyType{field_id: 'value'}`.

  Fields:
    entries: The entries in the creation expression.
    messageName: The type name of the message to be created, empty when
      creating map literals.
  """
    entries = _messages.MessageField('GoogleApiExprExprCreateStructEntry', 1, repeated=True)
    messageName = _messages.StringField(2)