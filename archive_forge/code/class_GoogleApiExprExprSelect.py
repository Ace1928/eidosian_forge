from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprSelect(_messages.Message):
    """A field selection expression. e.g. `request.auth`.

  Fields:
    field: Required. The name of the field to select. For example, in the
      select expression `request.auth`, the `auth` portion of the expression
      would be the `field`.
    operand: Required. The target of the selection expression. For example, in
      the select expression `request.auth`, the `request` portion of the
      expression is the `operand`.
    testOnly: Whether the select is to be interpreted as a field presence
      test. This results from the macro `has(request.auth)`.
  """
    field = _messages.StringField(1)
    operand = _messages.MessageField('GoogleApiExprExpr', 2)
    testOnly = _messages.BooleanField(3)