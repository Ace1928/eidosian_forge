from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CpuOvercommitTypeValueValuesEnum(_messages.Enum):
    """CPU overcommit.

    Values:
      CPU_OVERCOMMIT_TYPE_UNSPECIFIED: <no description>
      ENABLED: <no description>
      NONE: <no description>
    """
    CPU_OVERCOMMIT_TYPE_UNSPECIFIED = 0
    ENABLED = 1
    NONE = 2