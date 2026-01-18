from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemplateTypeValueValuesEnum(_messages.Enum):
    """Template Type.

    Values:
      UNKNOWN: Unknown Template Type.
      LEGACY: Legacy Template.
      FLEX: Flex Template.
    """
    UNKNOWN = 0
    LEGACY = 1
    FLEX = 2