from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespacedResourceRestoreModeValueValuesEnum(_messages.Enum):
    """Optional. Defines the behavior for handling the situation where sets
    of namespaced resources being restored already exist in the target
    cluster. This MUST be set to a value other than
    NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED.

    Values:
      NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED: Unspecified (invalid).
      DELETE_AND_RESTORE: When conflicting top-level resources (either
        Namespaces or ProtectedApplications, depending upon the scope) are
        encountered, this will first trigger a delete of the conflicting
        resource AND ALL OF ITS REFERENCED RESOURCES (e.g., all resources in
        the Namespace or all resources referenced by the ProtectedApplication)
        before restoring the resources from the Backup. This mode should only
        be used when you are intending to revert some portion of a cluster to
        an earlier state.
      FAIL_ON_CONFLICT: If conflicting top-level resources (either Namespaces
        or ProtectedApplications, depending upon the scope) are encountered at
        the beginning of a restore process, the Restore will fail. If a
        conflict occurs during the restore process itself (e.g., because an
        out of band process creates conflicting resources), a conflict will be
        reported.
      MERGE_SKIP_ON_CONFLICT: This mode merges the backup and the target
        cluster and skips the conflicting resources. If a single resource to
        restore exists in the cluster before restoration, the resource will be
        skipped, otherwise it will be restored.
      MERGE_REPLACE_VOLUME_ON_CONFLICT: This mode merges the backup and the
        target cluster and skips the conflicting resources except volume data.
        If a PVC to restore already exists, this mode will restore/reconnect
        the volume without overwriting the PVC. It is similar to
        MERGE_SKIP_ON_CONFLICT except that it will apply the volume data
        policy for the conflicting PVCs: - RESTORE_VOLUME_DATA_FROM_BACKUP:
        restore data only and respect the reclaim policy of the original PV; -
        REUSE_VOLUME_HANDLE_FROM_BACKUP: reconnect and respect the reclaim
        policy of the original PV; - NO_VOLUME_DATA_RESTORATION: new provision
        and respect the reclaim policy of the original PV. Note that this mode
        could cause data loss as the original PV can be retained or deleted
        depending on its reclaim policy.
      MERGE_REPLACE_ON_CONFLICT: This mode merges the backup and the target
        cluster and replaces the conflicting resources with the ones in the
        backup. If a single resource to restore exists in the cluster before
        restoration, the resource will be replaced with the one from the
        backup. To replace an existing resource, the first attempt is to
        update the resource to match the one from the backup; if the update
        fails, the second attempt is to delete the resource and restore it
        from the backup. Note that this mode could cause data loss as it
        replaces the existing resources in the target cluster, and the
        original PV can be retained or deleted depending on its reclaim
        policy.
    """
    NAMESPACED_RESOURCE_RESTORE_MODE_UNSPECIFIED = 0
    DELETE_AND_RESTORE = 1
    FAIL_ON_CONFLICT = 2
    MERGE_SKIP_ON_CONFLICT = 3
    MERGE_REPLACE_VOLUME_ON_CONFLICT = 4
    MERGE_REPLACE_ON_CONFLICT = 5