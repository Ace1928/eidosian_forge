from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementValueValuesEnum(_messages.Enum):
    """Enables automatic Service Mesh management.

    Values:
      MANAGEMENT_UNSPECIFIED: Unspecified
      MANAGEMENT_AUTOMATIC: Google should manage my Service Mesh for the
        cluster.
      MANAGEMENT_MANUAL: User will manually configure their service mesh
        components.
    """
    MANAGEMENT_UNSPECIFIED = 0
    MANAGEMENT_AUTOMATIC = 1
    MANAGEMENT_MANUAL = 2