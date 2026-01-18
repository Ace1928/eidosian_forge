from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AvailabilityValueValuesEnum(_messages.Enum):
    """Output only. Availability of the instance.

    Values:
      UNSPECIFIED: <no description>
      RESIDENT: <no description>
      DYNAMIC: <no description>
    """
    UNSPECIFIED = 0
    RESIDENT = 1
    DYNAMIC = 2