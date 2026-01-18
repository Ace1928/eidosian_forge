from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExprIdent(_messages.Message):
    """An identifier expression. e.g. `request`.

  Fields:
    name: Required. Holds a single, unqualified identifier, possibly preceded
      by a '.'. Qualified names are represented by the Expr.Select expression.
  """
    name = _messages.StringField(1)