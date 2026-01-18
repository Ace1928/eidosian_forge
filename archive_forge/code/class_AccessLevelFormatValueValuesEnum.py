from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessLevelFormatValueValuesEnum(_messages.Enum):
    """Whether to return `BasicLevels` in the Cloud Common Expression
    language, as `CustomLevels`, rather than as `BasicLevels`. Defaults to
    returning `AccessLevels` in the format they were defined.

    Values:
      LEVEL_FORMAT_UNSPECIFIED: The format was not specified.
      AS_DEFINED: Uses the format the resource was defined in. BasicLevels are
        returned as BasicLevels, CustomLevels are returned as CustomLevels.
      CEL: Use Cloud Common Expression Language when returning the resource.
        Both BasicLevels and CustomLevels are returned as CustomLevels.
    """
    LEVEL_FORMAT_UNSPECIFIED = 0
    AS_DEFINED = 1
    CEL = 2