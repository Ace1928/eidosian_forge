from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceBackupProperties(_messages.Message):
    """ComputeInstanceBackupProperties represents Compute Engine instance
  backup properties.

  Enums:
    KeyRevocationActionTypeValueValuesEnum: KeyRevocationActionType of the
      instance. Supported options are "STOP" and "NONE". The default value is
      "NONE" if it is not specified.

  Fields:
    canIpForward: Enables instances created based on these properties to send
      packets with source IP addresses other than their own and receive
      packets with destination IP addresses other than their own. If these
      instances will be used as an IP gateway or it will be set as the next-
      hop in a Route resource, specify `true`. If unsure, leave this set to
      `false`. See the https://cloud.google.com/vpc/docs/using-
      routes#canipforward documentation for more information.
    description: An optional text description for the instances that are
      created from these properties.
    disk: An array of disks that are associated with the instances that are
      created from these properties.
    guestAccelerator: A list of guest accelerator cards' type and count to use
      for instances created from these properties.
    keyRevocationActionType: KeyRevocationActionType of the instance.
      Supported options are "STOP" and "NONE". The default value is "NONE" if
      it is not specified.
    machineType: The machine type to use for instances that are created from
      these properties.
    metadata: The metadata key/value pairs to assign to instances that are
      created from these properties. These pairs can consist of custom
      metadata or predefined keys. See
      https://cloud.google.com/compute/docs/metadata/overview for more
      information.
    minCpuPlatform: Minimum cpu/platform to be used by instances. The instance
      may be scheduled on the specified or newer cpu/platform. Applicable
      values are the friendly names of CPU platforms, such as `minCpuPlatform:
      Intel Haswell` or `minCpuPlatform: Intel Sandy Bridge`. For more
      information, read
      https://cloud.google.com/compute/docs/instances/specify-min-cpu-
      platform.
    networkInterface: An array of network access configurations for this
      interface.
    scheduling: Specifies the scheduling options for the instances that are
      created from these properties.
    serviceAccount: A list of service accounts with specified scopes. Access
      tokens for these service accounts are available to the instances that
      are created from these properties. Use metadata queries to obtain the
      access tokens for these instances.
    tags: A list of tags to apply to the instances that are created from these
      properties. The tags identify valid sources or targets for network
      firewalls. The setTags method can modify this list of tags. Each tag
      within the list must comply with RFC1035
      (https://www.ietf.org/rfc/rfc1035.txt).
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
    canIpForward = _messages.BooleanField(1)
    description = _messages.StringField(2)
    disk = _messages.MessageField('AttachedDisk', 3, repeated=True)
    guestAccelerator = _messages.MessageField('AcceleratorConfig', 4, repeated=True)
    keyRevocationActionType = _messages.EnumField('KeyRevocationActionTypeValueValuesEnum', 5)
    machineType = _messages.StringField(6)
    metadata = _messages.MessageField('Metadata', 7)
    minCpuPlatform = _messages.StringField(8)
    networkInterface = _messages.MessageField('NetworkInterface', 9, repeated=True)
    scheduling = _messages.MessageField('Scheduling', 10)
    serviceAccount = _messages.MessageField('ServiceAccount', 11, repeated=True)
    tags = _messages.MessageField('Tags', 12)