from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroup(_messages.Message):
    """Represents a collection of network endpoints. A network endpoint group
  (NEG) defines how a set of endpoints should be reached, whether they are
  reachable, and where they are located. For more information about using NEGs
  for different use cases, see Network endpoint groups overview.

  Enums:
    ClientPortMappingModeValueValuesEnum: Only valid when networkEndpointType
      is GCE_VM_IP_PORT and the NEG is regional.
    NetworkEndpointTypeValueValuesEnum: Type of network endpoints in this
      network endpoint group. Can be one of GCE_VM_IP, GCE_VM_IP_PORT,
      NON_GCP_PRIVATE_IP_PORT, INTERNET_FQDN_PORT, INTERNET_IP_PORT,
      SERVERLESS, PRIVATE_SERVICE_CONNECT, GCE_VM_IP_PORTMAP.

  Messages:
    AnnotationsValue: Metadata defined as annotations on the network endpoint
      group.

  Fields:
    annotations: Metadata defined as annotations on the network endpoint
      group.
    appEngine: Only valid when networkEndpointType is SERVERLESS. Only one of
      cloudRun, appEngine or cloudFunction may be set.
    clientPortMappingMode: Only valid when networkEndpointType is
      GCE_VM_IP_PORT and the NEG is regional.
    cloudFunction: Only valid when networkEndpointType is SERVERLESS. Only one
      of cloudRun, appEngine or cloudFunction may be set.
    cloudRun: Only valid when networkEndpointType is SERVERLESS. Only one of
      cloudRun, appEngine or cloudFunction may be set.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    defaultPort: The default port used if the port number is not specified in
      the network endpoint. If the network endpoint type is either GCE_VM_IP,
      SERVERLESS or PRIVATE_SERVICE_CONNECT, this field must not be specified.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always
      compute#networkEndpointGroup for network endpoint group.
    loadBalancer: This field is only valid when the network endpoint group is
      used for load balancing. [Deprecated] This field is deprecated.
    name: Name of the resource; provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    network: The URL of the network to which all network endpoints in the NEG
      belong. Uses default project network if unspecified.
    networkEndpointType: Type of network endpoints in this network endpoint
      group. Can be one of GCE_VM_IP, GCE_VM_IP_PORT, NON_GCP_PRIVATE_IP_PORT,
      INTERNET_FQDN_PORT, INTERNET_IP_PORT, SERVERLESS,
      PRIVATE_SERVICE_CONNECT, GCE_VM_IP_PORTMAP.
    pscData: A NetworkEndpointGroupPscData attribute.
    pscTargetService: The target service url used to set up private service
      connection to a Google API or a PSC Producer Service Attachment. An
      example value is: asia-northeast3-cloudkms.googleapis.com
    region: [Output Only] The URL of the region where the network endpoint
      group is located.
    selfLink: [Output Only] Server-defined URL for the resource.
    serverlessDeployment: Only valid when networkEndpointType is SERVERLESS.
      Only one of cloudRun, appEngine, cloudFunction or serverlessDeployment
      may be set.
    size: [Output only] Number of network endpoints in the network endpoint
      group.
    subnetwork: Optional URL of the subnetwork to which all network endpoints
      in the NEG belong.
    zone: [Output Only] The URL of the zone where the network endpoint group
      is located.
  """

    class ClientPortMappingModeValueValuesEnum(_messages.Enum):
        """Only valid when networkEndpointType is GCE_VM_IP_PORT and the NEG is
    regional.

    Values:
      CLIENT_PORT_PER_ENDPOINT: For each endpoint there is exactly one client
        port.
      PORT_MAPPING_DISABLED: NEG should not be used for mapping client port to
        destination.
    """
        CLIENT_PORT_PER_ENDPOINT = 0
        PORT_MAPPING_DISABLED = 1

    class NetworkEndpointTypeValueValuesEnum(_messages.Enum):
        """Type of network endpoints in this network endpoint group. Can be one
    of GCE_VM_IP, GCE_VM_IP_PORT, NON_GCP_PRIVATE_IP_PORT, INTERNET_FQDN_PORT,
    INTERNET_IP_PORT, SERVERLESS, PRIVATE_SERVICE_CONNECT, GCE_VM_IP_PORTMAP.

    Values:
      GCE_VM_IP: The network endpoint is represented by an IP address.
      GCE_VM_IP_PORT: The network endpoint is represented by IP address and
        port pair.
      GCE_VM_IP_PORTMAP: The network endpoint is represented by an IP, Port
        and Client Destination Port.
      INTERNET_FQDN_PORT: The network endpoint is represented by fully
        qualified domain name and port.
      INTERNET_IP_PORT: The network endpoint is represented by an internet IP
        address and port.
      NON_GCP_PRIVATE_IP_PORT: The network endpoint is represented by an IP
        address and port. The endpoint belongs to a VM or pod running in a
        customer's on-premises.
      PRIVATE_SERVICE_CONNECT: The network endpoint is either public Google
        APIs or services exposed by other GCP Project with a Service
        Attachment. The connection is set up by private service connect
      SERVERLESS: The network endpoint is handled by specified serverless
        infrastructure.
    """
        GCE_VM_IP = 0
        GCE_VM_IP_PORT = 1
        GCE_VM_IP_PORTMAP = 2
        INTERNET_FQDN_PORT = 3
        INTERNET_IP_PORT = 4
        NON_GCP_PRIVATE_IP_PORT = 5
        PRIVATE_SERVICE_CONNECT = 6
        SERVERLESS = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Metadata defined as annotations on the network endpoint group.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    appEngine = _messages.MessageField('NetworkEndpointGroupAppEngine', 2)
    clientPortMappingMode = _messages.EnumField('ClientPortMappingModeValueValuesEnum', 3)
    cloudFunction = _messages.MessageField('NetworkEndpointGroupCloudFunction', 4)
    cloudRun = _messages.MessageField('NetworkEndpointGroupCloudRun', 5)
    creationTimestamp = _messages.StringField(6)
    defaultPort = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    description = _messages.StringField(8)
    id = _messages.IntegerField(9, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(10, default='compute#networkEndpointGroup')
    loadBalancer = _messages.MessageField('NetworkEndpointGroupLbNetworkEndpointGroup', 11)
    name = _messages.StringField(12)
    network = _messages.StringField(13)
    networkEndpointType = _messages.EnumField('NetworkEndpointTypeValueValuesEnum', 14)
    pscData = _messages.MessageField('NetworkEndpointGroupPscData', 15)
    pscTargetService = _messages.StringField(16)
    region = _messages.StringField(17)
    selfLink = _messages.StringField(18)
    serverlessDeployment = _messages.MessageField('NetworkEndpointGroupServerlessDeployment', 19)
    size = _messages.IntegerField(20, variant=_messages.Variant.INT32)
    subnetwork = _messages.StringField(21)
    zone = _messages.StringField(22)