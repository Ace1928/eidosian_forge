from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateConnection(_messages.Message):
    """Private connection resource that provides connectivity for VMware Engine
  private clouds.

  Enums:
    PeeringStateValueValuesEnum: Output only. Peering state between service
      network and VMware Engine network.
    RoutingModeValueValuesEnum: Optional. Routing Mode. Default value is set
      to GLOBAL. For type = PRIVATE_SERVICE_ACCESS, this field can be set to
      GLOBAL or REGIONAL, for other types only GLOBAL is supported.
    StateValueValuesEnum: Output only. State of the private connection.
    TypeValueValuesEnum: Required. Private connection type.

  Fields:
    createTime: Output only. Creation time of this resource.
    description: Optional. User-provided description for this private
      connection.
    name: Output only. The resource name of the private connection. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/privateConnections/my-
      connection`
    peeringId: Output only. VPC network peering id between given network VPC
      and VMwareEngineNetwork.
    peeringState: Output only. Peering state between service network and
      VMware Engine network.
    routingMode: Optional. Routing Mode. Default value is set to GLOBAL. For
      type = PRIVATE_SERVICE_ACCESS, this field can be set to GLOBAL or
      REGIONAL, for other types only GLOBAL is supported.
    serviceNetwork: Required. Service network to create private connection.
      Specify the name in the following form:
      `projects/{project}/global/networks/{network_id}` For type =
      PRIVATE_SERVICE_ACCESS, this field represents servicenetworking VPC,
      e.g. projects/project-tp/global/networks/servicenetworking. For type =
      NETAPP_CLOUD_VOLUME, this field represents NetApp service VPC, e.g.
      projects/project-tp/global/networks/netapp-tenant-vpc. For type =
      DELL_POWERSCALE, this field represent Dell service VPC, e.g.
      projects/project-tp/global/networks/dell-tenant-vpc. For type=
      THIRD_PARTY_SERVICE, this field could represent a consumer VPC or any
      other producer VPC to which the VMware Engine Network needs to be
      connected, e.g. projects/project/global/networks/vpc.
    state: Output only. State of the private connection.
    type: Required. Private connection type.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
    vmwareEngineNetwork: Required. The relative resource name of Legacy VMware
      Engine network. Specify the name in the following form: `projects/{proje
      ct}/locations/{location}/vmwareEngineNetworks/{vmware_engine_network_id}
      ` where `{project}`, `{location}` will be same as specified in private
      connection resource name and `{vmware_engine_network_id}` will be in the
      form of `{location}`-default e.g. projects/project/locations/us-
      central1/vmwareEngineNetworks/us-central1-default.
    vmwareEngineNetworkCanonical: Output only. The canonical name of the
      VMware Engine network in the form: `projects/{project_number}/locations/
      {location}/vmwareEngineNetworks/{vmware_engine_network_id}`
  """

    class PeeringStateValueValuesEnum(_messages.Enum):
        """Output only. Peering state between service network and VMware Engine
    network.

    Values:
      PEERING_STATE_UNSPECIFIED: The default value. This value is used if the
        peering state is omitted or unknown.
      PEERING_ACTIVE: The peering is in active state.
      PEERING_INACTIVE: The peering is in inactive state.
    """
        PEERING_STATE_UNSPECIFIED = 0
        PEERING_ACTIVE = 1
        PEERING_INACTIVE = 2

    class RoutingModeValueValuesEnum(_messages.Enum):
        """Optional. Routing Mode. Default value is set to GLOBAL. For type =
    PRIVATE_SERVICE_ACCESS, this field can be set to GLOBAL or REGIONAL, for
    other types only GLOBAL is supported.

    Values:
      ROUTING_MODE_UNSPECIFIED: The default value. This value should never be
        used.
      GLOBAL: Global Routing Mode
      REGIONAL: Regional Routing Mode
    """
        ROUTING_MODE_UNSPECIFIED = 0
        GLOBAL = 1
        REGIONAL = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the private connection.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      CREATING: The private connection is being created.
      ACTIVE: The private connection is ready.
      UPDATING: The private connection is being updated.
      DELETING: The private connection is being deleted.
      UNPROVISIONED: The private connection is not provisioned, since no
        private cloud is present for which this private connection is needed.
      FAILED: The private connection is in failed state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        UPDATING = 3
        DELETING = 4
        UNPROVISIONED = 5
        FAILED = 6

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Private connection type.

    Values:
      TYPE_UNSPECIFIED: The default value. This value should never be used.
      PRIVATE_SERVICE_ACCESS: Connection used for establishing [private
        services access](https://cloud.google.com/vpc/docs/private-services-
        access).
      NETAPP_CLOUD_VOLUMES: Connection used for connecting to NetApp Cloud
        Volumes.
      DELL_POWERSCALE: Connection used for connecting to Dell PowerScale.
      THIRD_PARTY_SERVICE: Connection used for connecting to third-party
        services.
      GOOGLE_CLOUD_NETAPP_VOLUMES: Connection used for connecting to Google
        Cloud NetApp Volumes.
    """
        TYPE_UNSPECIFIED = 0
        PRIVATE_SERVICE_ACCESS = 1
        NETAPP_CLOUD_VOLUMES = 2
        DELL_POWERSCALE = 3
        THIRD_PARTY_SERVICE = 4
        GOOGLE_CLOUD_NETAPP_VOLUMES = 5
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    name = _messages.StringField(3)
    peeringId = _messages.StringField(4)
    peeringState = _messages.EnumField('PeeringStateValueValuesEnum', 5)
    routingMode = _messages.EnumField('RoutingModeValueValuesEnum', 6)
    serviceNetwork = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    type = _messages.EnumField('TypeValueValuesEnum', 9)
    uid = _messages.StringField(10)
    updateTime = _messages.StringField(11)
    vmwareEngineNetwork = _messages.StringField(12)
    vmwareEngineNetworkCanonical = _messages.StringField(13)