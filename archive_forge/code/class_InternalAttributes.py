from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InternalAttributes(_messages.Message):
    """Attributes associated with workload.

  Enums:
    ManagerTypeValueValuesEnum: Output only. The GCP resource/product
      responsible for this workload.

  Fields:
    managedRegistration: Output only. Defines if Workload is managed.
    managerType: Output only. The GCP resource/product responsible for this
      workload.
  """

    class ManagerTypeValueValuesEnum(_messages.Enum):
        """Output only. The GCP resource/product responsible for this workload.

    Values:
      TYPE_UNSPECIFIED: Default. Should not be used.
      GKE_HUB: Resource managed by GKE Hub.
      BACKEND_SERVICE: Resource managed by Arcus, Backend Service
    """
        TYPE_UNSPECIFIED = 0
        GKE_HUB = 1
        BACKEND_SERVICE = 2
    managedRegistration = _messages.BooleanField(1)
    managerType = _messages.EnumField('ManagerTypeValueValuesEnum', 2)