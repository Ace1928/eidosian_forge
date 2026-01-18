from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetInstance(_messages.Message):
    """Represents a Target Instance resource. You can use a target instance to
  handle traffic for one or more forwarding rules, which is ideal for
  forwarding protocol traffic that is managed by a single source. For example,
  ESP, AH, TCP, or UDP. For more information, read Target instances.

  Enums:
    NatPolicyValueValuesEnum: Must have a value of NO_NAT. Protocol forwarding
      delivers packets while preserving the destination IP address of the
      forwarding rule referencing the target instance.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    instance: A URL to the virtual machine instance that handles traffic for
      this target instance. When creating a target instance, you can provide
      the fully-qualified URL or a valid partial URL to the desired virtual
      machine. For example, the following are all valid URLs: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /instances/instance - projects/project/zones/zone/instances/instance -
      zones/zone/instances/instance
    kind: [Output Only] The type of the resource. Always
      compute#targetInstance for target instances.
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    natPolicy: Must have a value of NO_NAT. Protocol forwarding delivers
      packets while preserving the destination IP address of the forwarding
      rule referencing the target instance.
    network: The URL of the network this target instance uses to forward
      traffic. If not specified, the traffic will be forwarded to the network
      that the default network interface belongs to.
    securityPolicy: [Output Only] The resource URL for the security policy
      associated with this target instance.
    selfLink: [Output Only] Server-defined URL for the resource.
    zone: [Output Only] URL of the zone where the target instance resides. You
      must specify this field as part of the HTTP request URL. It is not
      settable as a field in the request body.
  """

    class NatPolicyValueValuesEnum(_messages.Enum):
        """Must have a value of NO_NAT. Protocol forwarding delivers packets
    while preserving the destination IP address of the forwarding rule
    referencing the target instance.

    Values:
      NO_NAT: No NAT performed.
    """
        NO_NAT = 0
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    id = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    instance = _messages.StringField(4)
    kind = _messages.StringField(5, default='compute#targetInstance')
    name = _messages.StringField(6)
    natPolicy = _messages.EnumField('NatPolicyValueValuesEnum', 7)
    network = _messages.StringField(8)
    securityPolicy = _messages.StringField(9)
    selfLink = _messages.StringField(10)
    zone = _messages.StringField(11)