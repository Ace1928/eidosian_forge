from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VolumeDataRestorePolicyValueValuesEnum(_messages.Enum):
    """Optional. Specifies the mechanism to be used to restore volume data.
    Default: VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED (will be treated as
    NO_VOLUME_DATA_RESTORATION).

    Values:
      VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED: Unspecified (illegal).
      RESTORE_VOLUME_DATA_FROM_BACKUP: For each PVC to be restored, create a
        new underlying volume and PV from the corresponding VolumeBackup
        contained within the Backup.
      REUSE_VOLUME_HANDLE_FROM_BACKUP: For each PVC to be restored, attempt to
        reuse the original PV contained in the Backup (with its original
        underlying volume). This option is likely only usable when restoring a
        workload to its original cluster.
      NO_VOLUME_DATA_RESTORATION: For each PVC to be restored, create PVC
        without any particular action to restore data. In this case, the
        normal Kubernetes provisioning logic would kick in, and this would
        likely result in either dynamically provisioning blank PVs or binding
        to statically provisioned PVs.
    """
    VOLUME_DATA_RESTORE_POLICY_UNSPECIFIED = 0
    RESTORE_VOLUME_DATA_FROM_BACKUP = 1
    REUSE_VOLUME_HANDLE_FROM_BACKUP = 2
    NO_VOLUME_DATA_RESTORATION = 3