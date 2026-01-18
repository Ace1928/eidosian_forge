from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidationOptions(_messages.Message):
    """Options for how to validate and process properties on a resource.

  Enums:
    SchemaValidationValueValuesEnum: Customize how deployment manager will
      validate the resource against schema errors.
    UndeclaredPropertiesValueValuesEnum: Specify what to do with extra
      properties when executing a request.

  Fields:
    schemaValidation: Customize how deployment manager will validate the
      resource against schema errors.
    undeclaredProperties: Specify what to do with extra properties when
      executing a request.
  """

    class SchemaValidationValueValuesEnum(_messages.Enum):
        """Customize how deployment manager will validate the resource against
    schema errors.

    Values:
      UNKNOWN: <no description>
      IGNORE: Ignore schema failures.
      IGNORE_WITH_WARNINGS: Ignore schema failures but display them as
        warnings.
      FAIL: Fail the resource if the schema is not valid, this is the default
        behavior.
    """
        UNKNOWN = 0
        IGNORE = 1
        IGNORE_WITH_WARNINGS = 2
        FAIL = 3

    class UndeclaredPropertiesValueValuesEnum(_messages.Enum):
        """Specify what to do with extra properties when executing a request.

    Values:
      UNKNOWN: <no description>
      INCLUDE: Always include even if not present on discovery doc.
      IGNORE: Always ignore if not present on discovery doc.
      INCLUDE_WITH_WARNINGS: Include on request, but emit a warning.
      IGNORE_WITH_WARNINGS: Ignore properties, but emit a warning.
      FAIL: Always fail if undeclared properties are present.
    """
        UNKNOWN = 0
        INCLUDE = 1
        IGNORE = 2
        INCLUDE_WITH_WARNINGS = 3
        IGNORE_WITH_WARNINGS = 4
        FAIL = 5
    schemaValidation = _messages.EnumField('SchemaValidationValueValuesEnum', 1)
    undeclaredProperties = _messages.EnumField('UndeclaredPropertiesValueValuesEnum', 2)