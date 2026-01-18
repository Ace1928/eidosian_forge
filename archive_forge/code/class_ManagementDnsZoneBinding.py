from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementDnsZoneBinding(_messages.Message):
    """Represents a binding between a network and the management DNS zone. A
  management DNS zone is the Cloud DNS cross-project binding zone that VMware
  Engine creates for each private cloud. It contains FQDNs and corresponding
  IP addresses for the private cloud's ESXi hosts and management VM appliances
  like vCenter and NSX Manager.

  Enums:
    StateValueValuesEnum: Output only. The state of the resource.

  Fields:
    createTime: Output only. Creation time of this resource.
    description: User-provided description for this resource.
    etag: Checksum that may be sent on update and delete requests to ensure
      that the user-provided value is up to date before the server processes a
      request. The server computes checksums based on the value of other
      fields in the request.
    name: Output only. The resource name of this binding. Resource names are
      schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/managementDnsZoneBindings/my-management-dns-zone-binding`
    state: Output only. The state of the resource.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
    vmwareEngineNetwork: Network to bind is a VMware Engine network. Specify
      the name in the following form for VMware engine network: `projects/{pro
      ject}/locations/global/vmwareEngineNetworks/{vmware_engine_network_id}`.
      `{project}` can either be a project number or a project ID.
    vpcNetwork: Network to bind is a standard consumer VPC. Specify the name
      in the following form for consumer VPC network:
      `projects/{project}/global/networks/{network_id}`. `{project}` can
      either be a project number or a project ID.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      ACTIVE: The binding is ready.
      CREATING: The binding is being created.
      UPDATING: The binding is being updated.
      DELETING: The binding is being deleted.
      FAILED: The binding has failed.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        UPDATING = 3
        DELETING = 4
        FAILED = 5
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    uid = _messages.StringField(6)
    updateTime = _messages.StringField(7)
    vmwareEngineNetwork = _messages.StringField(8)
    vpcNetwork = _messages.StringField(9)