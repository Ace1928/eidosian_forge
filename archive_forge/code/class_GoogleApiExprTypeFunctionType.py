from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprTypeFunctionType(_messages.Message):
    """Function type with result and arg types.

  Fields:
    argTypes: Argument types of the function.
    resultType: Result type of the function.
  """
    argTypes = _messages.MessageField('GoogleApiExprType', 1, repeated=True)
    resultType = _messages.MessageField('GoogleApiExprType', 2)