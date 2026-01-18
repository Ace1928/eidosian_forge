from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceProperties(_messages.Message):
    """A InstanceProperties object.

  Enums:
    KeyRevocationActionTypeValueValuesEnum: KeyRevocationActionType of the
      instance. Supported options are "STOP" and "NONE". The default value is
      "NONE" if it is not specified.
    PostKeyRevocationActionTypeValueValuesEnum: PostKeyRevocationActionType of
      the instance.
    PrivateIpv6GoogleAccessValueValuesEnum: The private IPv6 google access
      type for VMs. If not specified, use INHERIT_FROM_SUBNETWORK as default.
      Note that for MachineImage, this is not supported yet.

  Messages:
    LabelsValue: Labels to apply to instances that are created from these
      properties.
    PartnerMetadataValue: Partner Metadata assigned to the instance
      properties. A map from a subdomain (namespace) to entries map.
    ResourceManagerTagsValue: Resource manager tags to be bound to the
      instance. Tag keys and values have the same definition as resource
      manager tags. Keys must be in the format `tagKeys/{tag_key_id}`, and
      values are in the format `tagValues/456`. The field is ignored (both PUT
      & PATCH) when empty.

  Fields:
    advancedMachineFeatures: Controls for advanced machine-related behavior
      features. Note that for MachineImage, this is not supported yet.
    canIpForward: Enables instances created based on these properties to send
      packets with source IP addresses other than their own and receive
      packets with destination IP addresses other than their own. If these
      instances will be used as an IP gateway or it will be set as the next-
      hop in a Route resource, specify true. If unsure, leave this set to
      false. See the Enable IP forwarding documentation for more information.
    confidentialInstanceConfig: Specifies the Confidential Instance options.
      Note that for MachineImage, this is not supported yet.
    description: An optional text description for the instances that are
      created from these properties.
    disks: An array of disks that are associated with the instances that are
      created from these properties.
    displayDevice: Display Device properties to enable support for remote
      display products like: Teradici, VNC and TeamViewer Note that for
      MachineImage, this is not supported yet.
    guestAccelerators: A list of guest accelerator cards' type and count to
      use for instances created from these properties.
    keyRevocationActionType: KeyRevocationActionType of the instance.
      Supported options are "STOP" and "NONE". The default value is "NONE" if
      it is not specified.
    labels: Labels to apply to instances that are created from these
      properties.
    machineType: The machine type to use for instances that are created from
      these properties.
    metadata: The metadata key/value pairs to assign to instances that are
      created from these properties. These pairs can consist of custom
      metadata or predefined keys. See Project and instance metadata for more
      information.
    minCpuPlatform: Minimum cpu/platform to be used by instances. The instance
      may be scheduled on the specified or newer cpu/platform. Applicable
      values are the friendly names of CPU platforms, such as minCpuPlatform:
      "Intel Haswell" or minCpuPlatform: "Intel Sandy Bridge". For more
      information, read Specifying a Minimum CPU Platform.
    networkInterfaces: An array of network access configurations for this
      interface.
    networkPerformanceConfig: Note that for MachineImage, this is not
      supported yet.
    partnerMetadata: Partner Metadata assigned to the instance properties. A
      map from a subdomain (namespace) to entries map.
    postKeyRevocationActionType: PostKeyRevocationActionType of the instance.
    privateIpv6GoogleAccess: The private IPv6 google access type for VMs. If
      not specified, use INHERIT_FROM_SUBNETWORK as default. Note that for
      MachineImage, this is not supported yet.
    reservationAffinity: Specifies the reservations that instances can consume
      from. Note that for MachineImage, this is not supported yet.
    resourceManagerTags: Resource manager tags to be bound to the instance.
      Tag keys and values have the same definition as resource manager tags.
      Keys must be in the format `tagKeys/{tag_key_id}`, and values are in the
      format `tagValues/456`. The field is ignored (both PUT & PATCH) when
      empty.
    resourcePolicies: Resource policies (names, not URLs) applied to instances
      created from these properties. Note that for MachineImage, this is not
      supported yet.
    scheduling: Specifies the scheduling options for the instances that are
      created from these properties.
    serviceAccounts: A list of service accounts with specified scopes. Access
      tokens for these service accounts are available to the instances that
      are created from these properties. Use metadata queries to obtain the
      access tokens for these instances.
    shieldedInstanceConfig: Note that for MachineImage, this is not supported
      yet.
    shieldedVmConfig: Specifies the Shielded VM options for the instances that
      are created from these properties.
    tags: A list of tags to apply to the instances that are created from these
      properties. The tags identify valid sources or targets for network
      firewalls. The setTags method can modify this list of tags. Each tag
      within the list must comply with RFC1035.
  """

    class KeyRevocationActionTypeValueValuesEnum(_messages.Enum):
        """KeyRevocationActionType of the instance. Supported options are "STOP"
    and "NONE". The default value is "NONE" if it is not specified.

    Values:
      KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED: Default value. This value is
        unused.
      NONE: Indicates user chose no operation.
      STOP: Indicates user chose to opt for VM shutdown on key revocation.
    """
        KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED = 0
        NONE = 1
        STOP = 2

    class PostKeyRevocationActionTypeValueValuesEnum(_messages.Enum):
        """PostKeyRevocationActionType of the instance.

    Values:
      NOOP: Indicates user chose no operation.
      POST_KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED: Default value. This value
        is unused.
      SHUTDOWN: Indicates user chose to opt for VM shutdown on key revocation.
    """
        NOOP = 0
        POST_KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED = 1
        SHUTDOWN = 2

    class PrivateIpv6GoogleAccessValueValuesEnum(_messages.Enum):
        """The private IPv6 google access type for VMs. If not specified, use
    INHERIT_FROM_SUBNETWORK as default. Note that for MachineImage, this is
    not supported yet.

    Values:
      ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE: Bidirectional private IPv6 access
        to/from Google services. If specified, the subnetwork who is attached
        to the instance's default network interface will be assigned an
        internal IPv6 prefix if it doesn't have before.
      ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE: Outbound private IPv6 access from
        VMs in this subnet to Google services. If specified, the subnetwork
        who is attached to the instance's default network interface will be
        assigned an internal IPv6 prefix if it doesn't have before.
      INHERIT_FROM_SUBNETWORK: Each network interface inherits
        PrivateIpv6GoogleAccess from its subnetwork.
    """
        ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE = 0
        ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE = 1
        INHERIT_FROM_SUBNETWORK = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels to apply to instances that are created from these properties.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PartnerMetadataValue(_messages.Message):
        """Partner Metadata assigned to the instance properties. A map from a
    subdomain (namespace) to entries map.

    Messages:
      AdditionalProperty: An additional property for a PartnerMetadataValue
        object.

    Fields:
      additionalProperties: Additional properties of type PartnerMetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PartnerMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A StructuredEntries attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('StructuredEntries', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceManagerTagsValue(_messages.Message):
        """Resource manager tags to be bound to the instance. Tag keys and values
    have the same definition as resource manager tags. Keys must be in the
    format `tagKeys/{tag_key_id}`, and values are in the format
    `tagValues/456`. The field is ignored (both PUT & PATCH) when empty.

    Messages:
      AdditionalProperty: An additional property for a
        ResourceManagerTagsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ResourceManagerTagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceManagerTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    advancedMachineFeatures = _messages.MessageField('AdvancedMachineFeatures', 1)
    canIpForward = _messages.BooleanField(2)
    confidentialInstanceConfig = _messages.MessageField('ConfidentialInstanceConfig', 3)
    description = _messages.StringField(4)
    disks = _messages.MessageField('AttachedDisk', 5, repeated=True)
    displayDevice = _messages.MessageField('DisplayDevice', 6)
    guestAccelerators = _messages.MessageField('AcceleratorConfig', 7, repeated=True)
    keyRevocationActionType = _messages.EnumField('KeyRevocationActionTypeValueValuesEnum', 8)
    labels = _messages.MessageField('LabelsValue', 9)
    machineType = _messages.StringField(10)
    metadata = _messages.MessageField('Metadata', 11)
    minCpuPlatform = _messages.StringField(12)
    networkInterfaces = _messages.MessageField('NetworkInterface', 13, repeated=True)
    networkPerformanceConfig = _messages.MessageField('NetworkPerformanceConfig', 14)
    partnerMetadata = _messages.MessageField('PartnerMetadataValue', 15)
    postKeyRevocationActionType = _messages.EnumField('PostKeyRevocationActionTypeValueValuesEnum', 16)
    privateIpv6GoogleAccess = _messages.EnumField('PrivateIpv6GoogleAccessValueValuesEnum', 17)
    reservationAffinity = _messages.MessageField('ReservationAffinity', 18)
    resourceManagerTags = _messages.MessageField('ResourceManagerTagsValue', 19)
    resourcePolicies = _messages.StringField(20, repeated=True)
    scheduling = _messages.MessageField('Scheduling', 21)
    serviceAccounts = _messages.MessageField('ServiceAccount', 22, repeated=True)
    shieldedInstanceConfig = _messages.MessageField('ShieldedInstanceConfig', 23)
    shieldedVmConfig = _messages.MessageField('ShieldedVmConfig', 24)
    tags = _messages.MessageField('Tags', 25)