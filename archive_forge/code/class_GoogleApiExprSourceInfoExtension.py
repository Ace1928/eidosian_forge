from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprSourceInfoExtension(_messages.Message):
    """An extension that was requested for the source expression.

  Enums:
    AffectedComponentsValueListEntryValuesEnum:

  Fields:
    affectedComponents: If set, the listed components must understand the
      extension for the expression to evaluate correctly. This field has set
      semantics, repeated values should be deduplicated.
    id: Identifier for the extension. Example: constant_folding
    version: Version info. May be skipped if it isn't meaningful for the
      extension. (for example constant_folding might always be v0.0).
  """

    class AffectedComponentsValueListEntryValuesEnum(_messages.Enum):
        """AffectedComponentsValueListEntryValuesEnum enum type.

    Values:
      COMPONENT_UNSPECIFIED: Unspecified, default.
      COMPONENT_PARSER: Parser. Converts a CEL string to an AST.
      COMPONENT_TYPE_CHECKER: Type checker. Checks that references in an AST
        are defined and types agree.
      COMPONENT_RUNTIME: Runtime. Evaluates a parsed and optionally checked
        CEL AST against a context.
    """
        COMPONENT_UNSPECIFIED = 0
        COMPONENT_PARSER = 1
        COMPONENT_TYPE_CHECKER = 2
        COMPONENT_RUNTIME = 3
    affectedComponents = _messages.EnumField('AffectedComponentsValueListEntryValuesEnum', 1, repeated=True)
    id = _messages.StringField(2)
    version = _messages.MessageField('GoogleApiExprSourceInfoExtensionVersion', 3)