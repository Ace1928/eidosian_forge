from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1CustomLevel(_messages.Message):
    """`CustomLevel` is an `AccessLevel` using the Cloud Common Expression
  Language to represent the necessary conditions for the level to apply to a
  request. See CEL spec at: https://github.com/google/cel-spec

  Fields:
    expr: Required. A Cloud CEL expression evaluating to a boolean.
  """
    expr = _messages.MessageField('Expr', 1)