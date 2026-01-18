from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Subnet(_messages.Message):
    """Subnet in a private cloud. Either `management` subnets (such as vMotion)
  that are read-only, or `userDefined`, which can also be updated.

  Enums:
    StateValueValuesEnum: Output only. The state of the resource.

  Fields:
    gatewayIp: The IP address of the gateway of this subnet. Must fall within
      the IP prefix defined above.
    ipCidrRange: The IP address range of the subnet in CIDR format
      '10.0.0.0/24'.
    name: Output only. The resource name of this subnet. Resource names are
      schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/subnets/my-subnet`
    state: Output only. The state of the resource.
    type: Output only. The type of the subnet. For example "management" or
      "userDefined".
    vlanId: Output only. VLAN ID of the VLAN on which the subnet is configured
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      ACTIVE: The subnet is ready.
      CREATING: The subnet is being created.
      UPDATING: The subnet is being updated.
      DELETING: The subnet is being deleted.
      RECONCILING: Changes requested in the last operation are being
        propagated.
      FAILED: Last operation on the subnet did not succeed. Subnet's payload
        is reverted back to its most recent working state.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        UPDATING = 3
        DELETING = 4
        RECONCILING = 5
        FAILED = 6
    gatewayIp = _messages.StringField(1)
    ipCidrRange = _messages.StringField(2)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    type = _messages.StringField(5)
    vlanId = _messages.IntegerField(6, variant=_messages.Variant.INT32)