from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareEngineNetwork(_messages.Message):
    """VMware Engine network resource that provides connectivity for VMware
  Engine private clouds.

  Enums:
    StateValueValuesEnum: Output only. State of the VMware Engine network.
    TypeValueValuesEnum: Required. VMware Engine network type.

  Fields:
    createTime: Output only. Creation time of this resource.
    description: User-provided description for this VMware Engine network.
    etag: Checksum that may be sent on update and delete requests to ensure
      that the user-provided value is up to date before the server processes a
      request. The server computes checksums based on the value of other
      fields in the request.
    name: Output only. The resource name of the VMware Engine network.
      Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/global/vmwareEngineNetworks/my-network`
    state: Output only. State of the VMware Engine network.
    type: Required. VMware Engine network type.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
    vpcNetworks: Output only. VMware Engine service VPC networks that provide
      connectivity from a private cloud to customer projects, the internet,
      and other Google Cloud services.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the VMware Engine network.

    Values:
      STATE_UNSPECIFIED: The default value. This value is used if the state is
        omitted.
      CREATING: The VMware Engine network is being created.
      ACTIVE: The VMware Engine network is ready.
      UPDATING: The VMware Engine network is being updated.
      DELETING: The VMware Engine network is being deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        UPDATING = 3
        DELETING = 4

    class TypeValueValuesEnum(_messages.Enum):
        """Required. VMware Engine network type.

    Values:
      TYPE_UNSPECIFIED: The default value. This value should never be used.
      LEGACY: Network type used by private clouds created in projects without
        a network of type `STANDARD`. This network type is no longer used for
        new VMware Engine private cloud deployments.
      STANDARD: Standard network type used for private cloud connectivity.
    """
        TYPE_UNSPECIFIED = 0
        LEGACY = 1
        STANDARD = 2
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    etag = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    type = _messages.EnumField('TypeValueValuesEnum', 6)
    uid = _messages.StringField(7)
    updateTime = _messages.StringField(8)
    vpcNetworks = _messages.MessageField('VpcNetwork', 9, repeated=True)