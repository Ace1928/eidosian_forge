from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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