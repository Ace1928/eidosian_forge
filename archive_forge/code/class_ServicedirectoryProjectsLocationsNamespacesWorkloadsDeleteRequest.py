from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesWorkloadsDeleteRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesWorkloadsDeleteRequest
  object.

  Enums:
    ManagerTypeValueValuesEnum: Stores extra information about what Google
      resource is directly responsible for a given Workload resource.

  Fields:
    managerType: Stores extra information about what Google resource is
      directly responsible for a given Workload resource.
    name: Required. The name of the workload to delete.
  """

    class ManagerTypeValueValuesEnum(_messages.Enum):
        """Stores extra information about what Google resource is directly
    responsible for a given Workload resource.

    Values:
      TYPE_UNSPECIFIED: Default. Should not be used.
      GKE_HUB: Resource managed by GKE Hub.
      BACKEND_SERVICE: Resource managed by Arcus, Backend Service
    """
        TYPE_UNSPECIFIED = 0
        GKE_HUB = 1
        BACKEND_SERVICE = 2
    managerType = _messages.EnumField('ManagerTypeValueValuesEnum', 1)
    name = _messages.StringField(2, required=True)