from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceInstanceProperties(_messages.Message):
    """DEPRECATED: Please use compute#instanceProperties instead. New
  properties will not be added to this field.

  Enums:
    KeyRevocationActionTypeValueValuesEnum: KeyRevocationActionType of the
      instance. Supported options are "STOP" and "NONE". The default value is
      "NONE" if it is not specified.
    PostKeyRevocationActionTypeValueValuesEnum: PostKeyRevocationActionType of
      the instance.

  Messages:
    LabelsValue: Labels to apply to instances that are created from this
      machine image.

  Fields:
    canIpForward: Enables instances created based on this machine image to
      send packets with source IP addresses other than their own and receive
      packets with destination IP addresses other than their own. If these
      instances will be used as an IP gateway or it will be set as the next-
      hop in a Route resource, specify true. If unsure, leave this set to
      false. See the Enable IP forwarding documentation for more information.
    deletionProtection: Whether the instance created from this machine image
      should be protected against deletion.
    description: An optional text description for the instances that are
      created from this machine image.
    disks: An array of disks that are associated with the instances that are
      created from this machine image.
    guestAccelerators: A list of guest accelerator cards' type and count to
      use for instances created from this machine image.
    keyRevocationActionType: KeyRevocationActionType of the instance.
      Supported options are "STOP" and "NONE". The default value is "NONE" if
      it is not specified.
    labels: Labels to apply to instances that are created from this machine
      image.
    machineType: The machine type to use for instances that are created from
      this machine image.
    metadata: The metadata key/value pairs to assign to instances that are
      created from this machine image. These pairs can consist of custom
      metadata or predefined keys. See Project and instance metadata for more
      information.
    minCpuPlatform: Minimum cpu/platform to be used by instances created from
      this machine image. The instance may be scheduled on the specified or
      newer cpu/platform. Applicable values are the friendly names of CPU
      platforms, such as minCpuPlatform: "Intel Haswell" or minCpuPlatform:
      "Intel Sandy Bridge". For more information, read Specifying a Minimum
      CPU Platform.
    networkInterfaces: An array of network access configurations for this
      interface.
    postKeyRevocationActionType: PostKeyRevocationActionType of the instance.
    scheduling: Specifies the scheduling options for the instances that are
      created from this machine image.
    serviceAccounts: A list of service accounts with specified scopes. Access
      tokens for these service accounts are available to the instances that
      are created from this machine image. Use metadata queries to obtain the
      access tokens for these instances.
    tags: A list of tags to apply to the instances that are created from this
      machine image. The tags identify valid sources or targets for network
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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels to apply to instances that are created from this machine image.

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
    canIpForward = _messages.BooleanField(1)
    deletionProtection = _messages.BooleanField(2)
    description = _messages.StringField(3)
    disks = _messages.MessageField('SavedAttachedDisk', 4, repeated=True)
    guestAccelerators = _messages.MessageField('AcceleratorConfig', 5, repeated=True)
    keyRevocationActionType = _messages.EnumField('KeyRevocationActionTypeValueValuesEnum', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    machineType = _messages.StringField(8)
    metadata = _messages.MessageField('Metadata', 9)
    minCpuPlatform = _messages.StringField(10)
    networkInterfaces = _messages.MessageField('NetworkInterface', 11, repeated=True)
    postKeyRevocationActionType = _messages.EnumField('PostKeyRevocationActionTypeValueValuesEnum', 12)
    scheduling = _messages.MessageField('Scheduling', 13)
    serviceAccounts = _messages.MessageField('ServiceAccount', 14, repeated=True)
    tags = _messages.MessageField('Tags', 15)