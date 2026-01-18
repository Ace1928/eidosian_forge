from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IncludeExtraDataValueValuesEnum(_messages.Enum):
    """Optional. Options to include extra data

    Values:
      INCLUDE_EXTRA_DATA_UNSPECIFIED: Unspecified: No extra data will be
        returned
      INCLUDE_EXTRA_DATA_NONE: Do not return any extra data
      INCLUDE_GOOGLE_GENERATED_METADATA: Return GGCM associated with the
        resources
    """
    INCLUDE_EXTRA_DATA_UNSPECIFIED = 0
    INCLUDE_EXTRA_DATA_NONE = 1
    INCLUDE_GOOGLE_GENERATED_METADATA = 2