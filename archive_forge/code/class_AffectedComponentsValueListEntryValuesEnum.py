from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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