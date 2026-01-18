from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeConfig(_messages.Message):
    """Configuration parameters for a new volume.

  Enums:
    PerformanceTierValueValuesEnum: Performance tier of the Volume. Default is
      SHARED.
    ProtocolValueValuesEnum: Volume protocol.
    TypeValueValuesEnum: The type of this Volume.

  Fields:
    gcpService: The GCP service of the storage volume. Available gcp_service
      are in https://cloud.google.com/bare-metal/docs/bms-planning.
    id: A transient unique identifier to identify a volume within an
      ProvisioningConfig request.
    lunRanges: LUN ranges to be configured. Set only when protocol is
      PROTOCOL_FC.
    machineIds: Machine ids connected to this volume. Set only when protocol
      is PROTOCOL_FC.
    name: Output only. The name of the volume config.
    nfsExports: NFS exports. Set only when protocol is PROTOCOL_NFS.
    performanceTier: Performance tier of the Volume. Default is SHARED.
    protocol: Volume protocol.
    sizeGb: The requested size of this volume, in GB.
    snapshotsEnabled: Whether snapshots should be enabled.
    storageAggregatePool: Input only. Name of the storage aggregate pool to
      allocate the volume in. Can be used only for
      VOLUME_PERFORMANCE_TIER_ASSIGNED volumes.
    type: The type of this Volume.
    userNote: User note field, it can be used by customers to add additional
      information for the BMS Ops team .
  """

    class PerformanceTierValueValuesEnum(_messages.Enum):
        """Performance tier of the Volume. Default is SHARED.

    Values:
      VOLUME_PERFORMANCE_TIER_UNSPECIFIED: Value is not specified.
      VOLUME_PERFORMANCE_TIER_SHARED: Regular volumes, shared aggregates.
      VOLUME_PERFORMANCE_TIER_ASSIGNED: Assigned aggregates.
      VOLUME_PERFORMANCE_TIER_HT: High throughput aggregates.
      VOLUME_PERFORMANCE_TIER_QOS2_PERFORMANCE: QoS 2.0 high performance
        storage.
    """
        VOLUME_PERFORMANCE_TIER_UNSPECIFIED = 0
        VOLUME_PERFORMANCE_TIER_SHARED = 1
        VOLUME_PERFORMANCE_TIER_ASSIGNED = 2
        VOLUME_PERFORMANCE_TIER_HT = 3
        VOLUME_PERFORMANCE_TIER_QOS2_PERFORMANCE = 4

    class ProtocolValueValuesEnum(_messages.Enum):
        """Volume protocol.

    Values:
      PROTOCOL_UNSPECIFIED: Unspecified value.
      PROTOCOL_FC: Fibre channel.
      PROTOCOL_NFS: Network file system.
    """
        PROTOCOL_UNSPECIFIED = 0
        PROTOCOL_FC = 1
        PROTOCOL_NFS = 2

    class TypeValueValuesEnum(_messages.Enum):
        """The type of this Volume.

    Values:
      TYPE_UNSPECIFIED: The unspecified type.
      FLASH: This Volume is on flash.
      DISK: This Volume is on disk.
    """
        TYPE_UNSPECIFIED = 0
        FLASH = 1
        DISK = 2
    gcpService = _messages.StringField(1)
    id = _messages.StringField(2)
    lunRanges = _messages.MessageField('LunRange', 3, repeated=True)
    machineIds = _messages.StringField(4, repeated=True)
    name = _messages.StringField(5)
    nfsExports = _messages.MessageField('NfsExport', 6, repeated=True)
    performanceTier = _messages.EnumField('PerformanceTierValueValuesEnum', 7)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 8)
    sizeGb = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    snapshotsEnabled = _messages.BooleanField(10)
    storageAggregatePool = _messages.StringField(11)
    type = _messages.EnumField('TypeValueValuesEnum', 12)
    userNote = _messages.StringField(13)