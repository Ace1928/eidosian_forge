from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshMembershipSpec(_messages.Message):
    """**Service Mesh**: Spec for a single Membership for the servicemesh
  feature

  Enums:
    ControlPlaneValueValuesEnum: Deprecated: use `management` instead Enables
      automatic control plane management.
    ManagementValueValuesEnum: Enables automatic Service Mesh management.

  Fields:
    controlPlane: Deprecated: use `management` instead Enables automatic
      control plane management.
    management: Enables automatic Service Mesh management.
  """

    class ControlPlaneValueValuesEnum(_messages.Enum):
        """Deprecated: use `management` instead Enables automatic control plane
    management.

    Values:
      CONTROL_PLANE_MANAGEMENT_UNSPECIFIED: Unspecified
      AUTOMATIC: Google should provision a control plane revision and make it
        available in the cluster. Google will enroll this revision in a
        release channel and keep it up to date. The control plane revision may
        be a managed service, or a managed install.
      MANUAL: User will manually configure the control plane (e.g. via CLI, or
        via the ControlPlaneRevision KRM API)
    """
        CONTROL_PLANE_MANAGEMENT_UNSPECIFIED = 0
        AUTOMATIC = 1
        MANUAL = 2

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
    controlPlane = _messages.EnumField('ControlPlaneValueValuesEnum', 1)
    management = _messages.EnumField('ManagementValueValuesEnum', 2)