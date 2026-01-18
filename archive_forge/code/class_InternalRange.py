from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InternalRange(_messages.Message):
    """The internal range resource for IPAM operations within a VPC network.
  Used to represent a private address range along with behavioral
  characterstics of that range (its usage and peering behavior). Networking
  resources can link to this range if they are created as belonging to it.

  Enums:
    OverlapsValueListEntryValuesEnum:
    PeeringValueValuesEnum: The type of peering set for this internal range.
    UsageValueValuesEnum: The type of usage set for this InternalRange.

  Messages:
    LabelsValue: User-defined labels.

  Fields:
    createTime: Time when the internal range was created.
    description: A description of this resource.
    ipCidrRange: The IP range that this internal range defines.
    labels: User-defined labels.
    name: Immutable. The name of an internal range. Format:
      projects/{project}/locations/{location}/internalRanges/{internal_range}
      See: https://google.aip.dev/122#fields-representing-resource-names
    network: The URL or resource ID of the network in which to reserve the
      internal range. The network cannot be deleted if there are any reserved
      internal ranges referring to it. Legacy networks are not supported. This
      can only be specified for a global internal address. Example: - URL:
      /compute/v1/projects/{project}/global/networks/{resourceId} - ID:
      network123
    overlaps: Optional. Types of resources that are allowed to overlap with
      the current internal range.
    peering: The type of peering set for this internal range.
    prefixLength: An alternate to ip_cidr_range. Can be set when trying to
      create a reservation that automatically finds a free range of the given
      size. If both ip_cidr_range and prefix_length are set, there is an error
      if the range sizes do not match. Can also be used during updates to
      change the range size.
    targetCidrRange: Optional. Can be set to narrow down or pick a different
      address space while searching for a free range. If not set, defaults to
      the "10.0.0.0/8" address space. This can be used to search in other
      rfc-1918 address spaces like "172.16.0.0/12" and "192.168.0.0/16" or
      non-rfc-1918 address spaces used in the VPC.
    updateTime: Time when the internal range was updated.
    usage: The type of usage set for this InternalRange.
    users: Output only. The list of resources that refer to this internal
      range. Resources that use the internal range for their range allocation
      are referred to as users of the range. Other resources mark themselves
      as users while doing so by creating a reference to this internal range.
      Having a user, based on this reference, prevents deletion of the
      internal range referred to. Can be empty.
  """

    class OverlapsValueListEntryValuesEnum(_messages.Enum):
        """OverlapsValueListEntryValuesEnum enum type.

    Values:
      OVERLAP_UNSPECIFIED: No overlap overrides.
      OVERLAP_ROUTE_RANGE: Allow creation of static routes more specific that
        the current internal range.
      OVERLAP_EXISTING_SUBNET_RANGE: Allow creation of internal ranges that
        overlap with existing subnets.
    """
        OVERLAP_UNSPECIFIED = 0
        OVERLAP_ROUTE_RANGE = 1
        OVERLAP_EXISTING_SUBNET_RANGE = 2

    class PeeringValueValuesEnum(_messages.Enum):
        """The type of peering set for this internal range.

    Values:
      PEERING_UNSPECIFIED: If Peering is left unspecified in
        CreateInternalRange or UpdateInternalRange, it will be defaulted to
        FOR_SELF.
      FOR_SELF: This is the default behavior and represents the case that this
        internal range is intended to be used in the VPC in which it is
        created and is accessible from its peers. This implies that peers or
        peers-of-peers cannot use this range.
      FOR_PEER: This behavior can be set when the internal range is being
        reserved for usage by peers. This means that no resource within the
        VPC in which it is being created can use this to associate with a VPC
        resource, but one of the peers can. This represents donating a range
        for peers to use.
      NOT_SHARED: This behavior can be set when the internal range is being
        reserved for usage by the VPC in which it is created, but not shared
        with peers. In a sense, it is local to the VPC. This can be used to
        create internal ranges for various purposes like
        HTTP_INTERNAL_LOAD_BALANCER or for Interconnect routes that are not
        shared with peers. This also implies that peers cannot use this range
        in a way that is visible to this VPC, but can re-use this range as
        long as it is NOT_SHARED from the peer VPC, too.
    """
        PEERING_UNSPECIFIED = 0
        FOR_SELF = 1
        FOR_PEER = 2
        NOT_SHARED = 3

    class UsageValueValuesEnum(_messages.Enum):
        """The type of usage set for this InternalRange.

    Values:
      USAGE_UNSPECIFIED: Unspecified usage is allowed in calls which identify
        the resource by other fields and do not need Usage set to complete.
        These are, i.e.: GetInternalRange and DeleteInternalRange. Usage needs
        to be specified explicitly in CreateInternalRange or
        UpdateInternalRange calls.
      FOR_VPC: A VPC resource can use the reserved CIDR block by associating
        it with the internal range resource if usage is set to FOR_VPC.
      EXTERNAL_TO_VPC: Ranges created with EXTERNAL_TO_VPC cannot be
        associated with VPC resources and are meant to block out address
        ranges for various use cases, like for example, usage on-prem, with
        dynamic route announcements via interconnect.
    """
        USAGE_UNSPECIFIED = 0
        FOR_VPC = 1
        EXTERNAL_TO_VPC = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels.

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
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    ipCidrRange = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    network = _messages.StringField(6)
    overlaps = _messages.EnumField('OverlapsValueListEntryValuesEnum', 7, repeated=True)
    peering = _messages.EnumField('PeeringValueValuesEnum', 8)
    prefixLength = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    targetCidrRange = _messages.StringField(10, repeated=True)
    updateTime = _messages.StringField(11)
    usage = _messages.EnumField('UsageValueValuesEnum', 12)
    users = _messages.StringField(13, repeated=True)