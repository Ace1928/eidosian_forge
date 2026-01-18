from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprCheckedExpr(_messages.Message):
    """A CEL expression which has been successfully type checked.

  Messages:
    ReferenceMapValue: A map from expression ids to resolved references. The
      following entries are in this table: - An Ident or Select expression is
      represented here if it resolves to a declaration. For instance, if
      `a.b.c` is represented by `select(select(id(a), b), c)`, and `a.b`
      resolves to a declaration, while `c` is a field selection, then the
      reference is attached to the nested select expression (but not to the id
      or or the outer select). In turn, if `a` resolves to a declaration and
      `b.c` are field selections, the reference is attached to the ident
      expression. - Every Call expression has an entry here, identifying the
      function being called. - Every CreateStruct expression for a message has
      an entry, identifying the message.
    TypeMapValue: A map from expression ids to types. Every expression node
      which has a type different than DYN has a mapping here. If an expression
      has type DYN, it is omitted from this map to save space.

  Fields:
    expr: The checked expression. Semantically equivalent to the parsed
      `expr`, but may have structural differences.
    exprVersion: The expr version indicates the major / minor version number
      of the `expr` representation. The most common reason for a version
      change will be to indicate to the CEL runtimes that transformations have
      been performed on the expr during static analysis. In some cases, this
      will save the runtime the work of applying the same or similar
      transformations prior to evaluation.
    referenceMap: A map from expression ids to resolved references. The
      following entries are in this table: - An Ident or Select expression is
      represented here if it resolves to a declaration. For instance, if
      `a.b.c` is represented by `select(select(id(a), b), c)`, and `a.b`
      resolves to a declaration, while `c` is a field selection, then the
      reference is attached to the nested select expression (but not to the id
      or or the outer select). In turn, if `a` resolves to a declaration and
      `b.c` are field selections, the reference is attached to the ident
      expression. - Every Call expression has an entry here, identifying the
      function being called. - Every CreateStruct expression for a message has
      an entry, identifying the message.
    sourceInfo: The source info derived from input that generated the parsed
      `expr` and any optimizations made during the type-checking pass.
    typeMap: A map from expression ids to types. Every expression node which
      has a type different than DYN has a mapping here. If an expression has
      type DYN, it is omitted from this map to save space.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ReferenceMapValue(_messages.Message):
        """A map from expression ids to resolved references. The following
    entries are in this table: - An Ident or Select expression is represented
    here if it resolves to a declaration. For instance, if `a.b.c` is
    represented by `select(select(id(a), b), c)`, and `a.b` resolves to a
    declaration, while `c` is a field selection, then the reference is
    attached to the nested select expression (but not to the id or or the
    outer select). In turn, if `a` resolves to a declaration and `b.c` are
    field selections, the reference is attached to the ident expression. -
    Every Call expression has an entry here, identifying the function being
    called. - Every CreateStruct expression for a message has an entry,
    identifying the message.

    Messages:
      AdditionalProperty: An additional property for a ReferenceMapValue
        object.

    Fields:
      additionalProperties: Additional properties of type ReferenceMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ReferenceMapValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiExprReference attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleApiExprReference', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TypeMapValue(_messages.Message):
        """A map from expression ids to types. Every expression node which has a
    type different than DYN has a mapping here. If an expression has type DYN,
    it is omitted from this map to save space.

    Messages:
      AdditionalProperty: An additional property for a TypeMapValue object.

    Fields:
      additionalProperties: Additional properties of type TypeMapValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TypeMapValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiExprType attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleApiExprType', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    expr = _messages.MessageField('GoogleApiExprExpr', 1)
    exprVersion = _messages.StringField(2)
    referenceMap = _messages.MessageField('ReferenceMapValue', 3)
    sourceInfo = _messages.MessageField('GoogleApiExprSourceInfo', 4)
    typeMap = _messages.MessageField('TypeMapValue', 5)