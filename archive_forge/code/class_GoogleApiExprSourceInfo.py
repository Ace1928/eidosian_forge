from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprSourceInfo(_messages.Message):
    """Source information collected at parse time.

  Messages:
    MacroCallsValue: A map from the parse node id where a macro replacement
      was made to the call `Expr` that resulted in a macro expansion. For
      example, `has(value.field)` is a function call that is replaced by a
      `test_only` field selection in the AST. Likewise, the call
      `list.exists(e, e > 10)` translates to a comprehension expression. The
      key in the map corresponds to the expression id of the expanded macro,
      and the value is the call `Expr` that was replaced.
    PositionsValue: A map from the parse node id (e.g. `Expr.id`) to the code
      point offset within the source.

  Fields:
    extensions: A list of tags for extensions that were used while parsing or
      type checking the source expression. For example, optimizations that
      require special runtime support may be specified. These are used to
      check feature support between components in separate implementations.
      This can be used to either skip redundant work or report an error if the
      extension is unsupported.
    lineOffsets: Monotonically increasing list of code point offsets where
      newlines `\\n` appear. The line number of a given position is the index
      `i` where for a given `id` the `line_offsets[i] < id_positions[id] <
      line_offsets[i+1]`. The column may be derived from `id_positions[id] -
      line_offsets[i]`.
    location: The location name. All position information attached to an
      expression is relative to this location. The location could be a file,
      UI element, or similar. For example, `acme/app/AnvilPolicy.cel`.
    macroCalls: A map from the parse node id where a macro replacement was
      made to the call `Expr` that resulted in a macro expansion. For example,
      `has(value.field)` is a function call that is replaced by a `test_only`
      field selection in the AST. Likewise, the call `list.exists(e, e > 10)`
      translates to a comprehension expression. The key in the map corresponds
      to the expression id of the expanded macro, and the value is the call
      `Expr` that was replaced.
    positions: A map from the parse node id (e.g. `Expr.id`) to the code point
      offset within the source.
    syntaxVersion: The syntax version of the source, e.g. `cel1`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MacroCallsValue(_messages.Message):
        """A map from the parse node id where a macro replacement was made to the
    call `Expr` that resulted in a macro expansion. For example,
    `has(value.field)` is a function call that is replaced by a `test_only`
    field selection in the AST. Likewise, the call `list.exists(e, e > 10)`
    translates to a comprehension expression. The key in the map corresponds
    to the expression id of the expanded macro, and the value is the call
    `Expr` that was replaced.

    Messages:
      AdditionalProperty: An additional property for a MacroCallsValue object.

    Fields:
      additionalProperties: Additional properties of type MacroCallsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MacroCallsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleApiExprExpr attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleApiExprExpr', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PositionsValue(_messages.Message):
        """A map from the parse node id (e.g. `Expr.id`) to the code point offset
    within the source.

    Messages:
      AdditionalProperty: An additional property for a PositionsValue object.

    Fields:
      additionalProperties: Additional properties of type PositionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PositionsValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    extensions = _messages.MessageField('GoogleApiExprSourceInfoExtension', 1, repeated=True)
    lineOffsets = _messages.IntegerField(2, repeated=True, variant=_messages.Variant.INT32)
    location = _messages.StringField(3)
    macroCalls = _messages.MessageField('MacroCallsValue', 4)
    positions = _messages.MessageField('PositionsValue', 5)
    syntaxVersion = _messages.StringField(6)