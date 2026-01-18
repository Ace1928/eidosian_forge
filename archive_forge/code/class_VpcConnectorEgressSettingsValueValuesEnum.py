from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpcConnectorEgressSettingsValueValuesEnum(_messages.Enum):
    """The egress settings for the connector, controlling what traffic is
    diverted through it.

    Values:
      VPC_CONNECTOR_EGRESS_SETTINGS_UNSPECIFIED: Unspecified.
      PRIVATE_RANGES_ONLY: Use the VPC Access Connector only for private IP
        space from RFC1918.
      ALL_TRAFFIC: Force the use of VPC Access Connector for all egress
        traffic from the function.
    """
    VPC_CONNECTOR_EGRESS_SETTINGS_UNSPECIFIED = 0
    PRIVATE_RANGES_ONLY = 1
    ALL_TRAFFIC = 2