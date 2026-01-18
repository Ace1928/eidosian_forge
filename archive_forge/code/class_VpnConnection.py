from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VpnConnection(_messages.Message):
    """A VPN connection .

  Enums:
    BgpRoutingModeValueValuesEnum: Dynamic routing mode of the VPC network,
      `regional` or `global`.

  Messages:
    LabelsValue: Labels associated with this resource.

  Fields:
    bgpRoutingMode: Dynamic routing mode of the VPC network, `regional` or
      `global`.
    cluster: The canonical Cluster name to connect to. It is in the form of
      projects/{project}/locations/{location}/clusters/{cluster}.
    createTime: Output only. The time when the VPN connection was created.
    details: Output only. The created connection details.
    enableHighAvailability: Whether this VPN connection has HA enabled on
      cluster side. If enabled, when creating VPN connection we will attempt
      to use 2 ANG floating IPs.
    labels: Labels associated with this resource.
    name: Required. The resource name of VPN connection
    natGatewayIp: NAT gateway IP, or WAN IP address. If a customer has
      multiple NAT IPs, the customer needs to configure NAT such that only one
      external IP maps to the GMEC Anthos cluster. This is empty if NAT is not
      used.
    router: Optional. The VPN connection Cloud Router name.
    updateTime: Output only. The time when the VPN connection was last
      updated.
    vpc: The network ID of VPC to connect to.
    vpcProject: Optional. Project detail of the VPC network. Required if VPC
      is in a different project than the cluster project.
  """

    class BgpRoutingModeValueValuesEnum(_messages.Enum):
        """Dynamic routing mode of the VPC network, `regional` or `global`.

    Values:
      BGP_ROUTING_MODE_UNSPECIFIED: Unknown.
      REGIONAL: Regional mode.
      GLOBAL: Global mode.
    """
        BGP_ROUTING_MODE_UNSPECIFIED = 0
        REGIONAL = 1
        GLOBAL = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with this resource.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    bgpRoutingMode = _messages.EnumField('BgpRoutingModeValueValuesEnum', 1)
    cluster = _messages.StringField(2)
    createTime = _messages.StringField(3)
    details = _messages.MessageField('Details', 4)
    enableHighAvailability = _messages.BooleanField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    natGatewayIp = _messages.StringField(8)
    router = _messages.StringField(9)
    updateTime = _messages.StringField(10)
    vpc = _messages.StringField(11)
    vpcProject = _messages.MessageField('VpcProject', 12)