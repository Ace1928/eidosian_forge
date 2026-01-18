from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprExpr(_messages.Message):
    """An abstract representation of a common expression. Expressions are
  abstractly represented as a collection of identifiers, select statements,
  function calls, literals, and comprehensions. All operators with the
  exception of the '.' operator are modelled as function calls. This makes it
  easy to represent new operators into the existing AST. All references within
  expressions must resolve to a Decl provided at type-check for an expression
  to be valid. A reference may either be a bare identifier `name` or a
  qualified identifier `google.api.name`. References may either refer to a
  value or a function declaration. For example, the expression
  `google.api.name.startsWith('expr')` references the declaration
  `google.api.name` within a Expr.Select expression, and the function
  declaration `startsWith`.

  Fields:
    callExpr: A call expression, including calls to predefined functions and
      operators.
    comprehensionExpr: A comprehension expression.
    constExpr: A constant expression.
    id: Required. An id assigned to this node by the parser which is unique in
      a given expression tree. This is used to associate type information and
      other attributes to a node in the parse tree.
    identExpr: An identifier expression.
    listExpr: A list creation expression.
    selectExpr: A field selection expression, e.g. `request.auth`.
    structExpr: A map or message creation expression.
  """
    callExpr = _messages.MessageField('GoogleApiExprExprCall', 1)
    comprehensionExpr = _messages.MessageField('GoogleApiExprExprComprehension', 2)
    constExpr = _messages.MessageField('GoogleApiExprConstant', 3)
    id = _messages.IntegerField(4)
    identExpr = _messages.MessageField('GoogleApiExprExprIdent', 5)
    listExpr = _messages.MessageField('GoogleApiExprExprCreateList', 6)
    selectExpr = _messages.MessageField('GoogleApiExprExprSelect', 7)
    structExpr = _messages.MessageField('GoogleApiExprExprCreateStruct', 8)