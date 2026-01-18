from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoExpansionModeValueValuesEnum(_messages.Enum):
    """Optional. Indicates whether the entity type can be automatically
    expanded.

    Values:
      AUTO_EXPANSION_MODE_UNSPECIFIED: Auto expansion disabled for the entity.
      AUTO_EXPANSION_MODE_DEFAULT: Allows an agent to recognize values that
        have not been explicitly listed in the entity.
    """
    AUTO_EXPANSION_MODE_UNSPECIFIED = 0
    AUTO_EXPANSION_MODE_DEFAULT = 1