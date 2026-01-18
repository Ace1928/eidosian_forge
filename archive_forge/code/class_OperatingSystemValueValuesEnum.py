from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperatingSystemValueValuesEnum(_messages.Enum):
    """Specifies the nodes operating system (default: LINUX).

    Values:
      OPERATING_SYSTEM_UNSPECIFIED: No operating system runtime selected.
      LINUX: Linux operating system.
    """
    OPERATING_SYSTEM_UNSPECIFIED = 0
    LINUX = 1