from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkConfig(_messages.Message):
    """Network configuration for ManagementServer instance.

  Enums:
    PeeringModeValueValuesEnum: Optional. The network connect mode of the
      ManagementServer instance. For this version, only PRIVATE_SERVICE_ACCESS
      is supported.

  Fields:
    network: Optional. The resource name of the Google Compute Engine VPC
      network to which the ManagementServer instance is connected.
    peeringMode: Optional. The network connect mode of the ManagementServer
      instance. For this version, only PRIVATE_SERVICE_ACCESS is supported.
  """

    class PeeringModeValueValuesEnum(_messages.Enum):
        """Optional. The network connect mode of the ManagementServer instance.
    For this version, only PRIVATE_SERVICE_ACCESS is supported.

    Values:
      PEERING_MODE_UNSPECIFIED: Peering mode not set.
      PRIVATE_SERVICE_ACCESS: Connect using Private Service Access to the
        Management Server. Private services access provides an IP address
        range for multiple Google Cloud services, including Cloud BackupDR.
    """
        PEERING_MODE_UNSPECIFIED = 0
        PRIVATE_SERVICE_ACCESS = 1
    network = _messages.StringField(1)
    peeringMode = _messages.EnumField('PeeringModeValueValuesEnum', 2)