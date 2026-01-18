from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataPlaneValueValuesEnum(_messages.Enum):
    """Enables automatic data plane management.

    Values:
      DATA_PLANE_MANAGEMENT_UNSPECIFIED: Unspecified
      DATA_PLANE_MANAGEMENT_AUTOMATIC: Enables Google-managed data plane that
        provides L7 service mesh capabilities. Data plane management is
        enabled at the cluster level. Users can exclude individual workloads
        or namespaces.
      DATA_PLANE_MANAGEMENT_MANUAL: User will manage their L7 data plane.
    """
    DATA_PLANE_MANAGEMENT_UNSPECIFIED = 0
    DATA_PLANE_MANAGEMENT_AUTOMATIC = 1
    DATA_PLANE_MANAGEMENT_MANUAL = 2