from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MachineType(_messages.Message):
    """Represents a Machine Type resource. You can use specific machine types
  for your VM instances based on performance and pricing requirements. For
  more information, read Machine Types.

  Messages:
    AcceleratorsValueListEntry: A AcceleratorsValueListEntry object.

  Fields:
    accelerators: [Output Only] A list of accelerator configurations assigned
      to this machine type.
    bundledLocalSsds: [Output Only] The configuration of bundled local SSD for
      the machine type.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    deprecated: [Output Only] The deprecation status associated with this
      machine type. Only applicable if the machine type is unavailable.
    description: [Output Only] An optional textual description of the
      resource.
    guestCpus: [Output Only] The number of virtual CPUs that are available to
      the instance.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    isSharedCpu: [Output Only] Whether this machine type has a shared CPU. See
      Shared-core machine types for more information.
    kind: [Output Only] The type of the resource. Always compute#machineType
      for machine types.
    maximumPersistentDisks: [Output Only] Maximum persistent disks allowed.
    maximumPersistentDisksSizeGb: [Output Only] Maximum total persistent disks
      size (GB) allowed.
    memoryMb: [Output Only] The amount of physical memory available to the
      instance, defined in MB.
    name: [Output Only] Name of the resource.
    selfLink: [Output Only] Server-defined URL for the resource.
    zone: [Output Only] The name of the zone where the machine type resides,
      such as us-central1-a.
  """

    class AcceleratorsValueListEntry(_messages.Message):
        """A AcceleratorsValueListEntry object.

    Fields:
      guestAcceleratorCount: Number of accelerator cards exposed to the guest.
      guestAcceleratorType: The accelerator type resource name, not a full
        URL, e.g. nvidia-tesla-t4.
    """
        guestAcceleratorCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
        guestAcceleratorType = _messages.StringField(2)
    accelerators = _messages.MessageField('AcceleratorsValueListEntry', 1, repeated=True)
    bundledLocalSsds = _messages.MessageField('BundledLocalSsds', 2)
    creationTimestamp = _messages.StringField(3)
    deprecated = _messages.MessageField('DeprecationStatus', 4)
    description = _messages.StringField(5)
    guestCpus = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    id = _messages.IntegerField(7, variant=_messages.Variant.UINT64)
    isSharedCpu = _messages.BooleanField(8)
    kind = _messages.StringField(9, default='compute#machineType')
    maximumPersistentDisks = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    maximumPersistentDisksSizeGb = _messages.IntegerField(11)
    memoryMb = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    name = _messages.StringField(13)
    selfLink = _messages.StringField(14)
    zone = _messages.StringField(15)