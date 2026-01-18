from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataOptions(_messages.Message):
    """Specifies the metadata options for running a transfer.

  Enums:
    AclValueValuesEnum: Specifies how each object's ACLs should be preserved
      for transfers between Google Cloud Storage buckets. If unspecified, the
      default behavior is the same as ACL_DESTINATION_BUCKET_DEFAULT.
    GidValueValuesEnum: Specifies how each file's POSIX group ID (GID)
      attribute should be handled by the transfer. By default, GID is not
      preserved. Only applicable to transfers involving POSIX file systems,
      and ignored for other transfers.
    KmsKeyValueValuesEnum: Specifies how each object's Cloud KMS customer-
      managed encryption key (CMEK) is preserved for transfers between Google
      Cloud Storage buckets. If unspecified, the default behavior is the same
      as KMS_KEY_DESTINATION_BUCKET_DEFAULT.
    ModeValueValuesEnum: Specifies how each file's mode attribute should be
      handled by the transfer. By default, mode is not preserved. Only
      applicable to transfers involving POSIX file systems, and ignored for
      other transfers.
    StorageClassValueValuesEnum: Specifies the storage class to set on objects
      being transferred to Google Cloud Storage buckets. If unspecified, the
      default behavior is the same as
      STORAGE_CLASS_DESTINATION_BUCKET_DEFAULT.
    SymlinkValueValuesEnum: Specifies how symlinks should be handled by the
      transfer. By default, symlinks are not preserved. Only applicable to
      transfers involving POSIX file systems, and ignored for other transfers.
    TemporaryHoldValueValuesEnum: Specifies how each object's temporary hold
      status should be preserved for transfers between Google Cloud Storage
      buckets. If unspecified, the default behavior is the same as
      TEMPORARY_HOLD_PRESERVE.
    TimeCreatedValueValuesEnum: Specifies how each object's `timeCreated`
      metadata is preserved for transfers. If unspecified, the default
      behavior is the same as TIME_CREATED_SKIP. This behavior is supported
      for transfers to GCS buckets from GCS, S3, Azure, S3 Compatible, and
      Azure sources.
    UidValueValuesEnum: Specifies how each file's POSIX user ID (UID)
      attribute should be handled by the transfer. By default, UID is not
      preserved. Only applicable to transfers involving POSIX file systems,
      and ignored for other transfers.

  Fields:
    acl: Specifies how each object's ACLs should be preserved for transfers
      between Google Cloud Storage buckets. If unspecified, the default
      behavior is the same as ACL_DESTINATION_BUCKET_DEFAULT.
    gid: Specifies how each file's POSIX group ID (GID) attribute should be
      handled by the transfer. By default, GID is not preserved. Only
      applicable to transfers involving POSIX file systems, and ignored for
      other transfers.
    kmsKey: Specifies how each object's Cloud KMS customer-managed encryption
      key (CMEK) is preserved for transfers between Google Cloud Storage
      buckets. If unspecified, the default behavior is the same as
      KMS_KEY_DESTINATION_BUCKET_DEFAULT.
    mode: Specifies how each file's mode attribute should be handled by the
      transfer. By default, mode is not preserved. Only applicable to
      transfers involving POSIX file systems, and ignored for other transfers.
    storageClass: Specifies the storage class to set on objects being
      transferred to Google Cloud Storage buckets. If unspecified, the default
      behavior is the same as STORAGE_CLASS_DESTINATION_BUCKET_DEFAULT.
    symlink: Specifies how symlinks should be handled by the transfer. By
      default, symlinks are not preserved. Only applicable to transfers
      involving POSIX file systems, and ignored for other transfers.
    temporaryHold: Specifies how each object's temporary hold status should be
      preserved for transfers between Google Cloud Storage buckets. If
      unspecified, the default behavior is the same as
      TEMPORARY_HOLD_PRESERVE.
    timeCreated: Specifies how each object's `timeCreated` metadata is
      preserved for transfers. If unspecified, the default behavior is the
      same as TIME_CREATED_SKIP. This behavior is supported for transfers to
      GCS buckets from GCS, S3, Azure, S3 Compatible, and Azure sources.
    uid: Specifies how each file's POSIX user ID (UID) attribute should be
      handled by the transfer. By default, UID is not preserved. Only
      applicable to transfers involving POSIX file systems, and ignored for
      other transfers.
  """

    class AclValueValuesEnum(_messages.Enum):
        """Specifies how each object's ACLs should be preserved for transfers
    between Google Cloud Storage buckets. If unspecified, the default behavior
    is the same as ACL_DESTINATION_BUCKET_DEFAULT.

    Values:
      ACL_UNSPECIFIED: ACL behavior is unspecified.
      ACL_DESTINATION_BUCKET_DEFAULT: Use the destination bucket's default
        object ACLS, if applicable.
      ACL_PRESERVE: Preserve the object's original ACLs. This requires the
        service account to have `storage.objects.getIamPolicy` permission for
        the source object. [Uniform bucket-level
        access](https://cloud.google.com/storage/docs/uniform-bucket-level-
        access) must not be enabled on either the source or destination
        buckets.
    """
        ACL_UNSPECIFIED = 0
        ACL_DESTINATION_BUCKET_DEFAULT = 1
        ACL_PRESERVE = 2

    class GidValueValuesEnum(_messages.Enum):
        """Specifies how each file's POSIX group ID (GID) attribute should be
    handled by the transfer. By default, GID is not preserved. Only applicable
    to transfers involving POSIX file systems, and ignored for other
    transfers.

    Values:
      GID_UNSPECIFIED: GID behavior is unspecified.
      GID_SKIP: Do not preserve GID during a transfer job.
      GID_NUMBER: Preserve GID during a transfer job.
    """
        GID_UNSPECIFIED = 0
        GID_SKIP = 1
        GID_NUMBER = 2

    class KmsKeyValueValuesEnum(_messages.Enum):
        """Specifies how each object's Cloud KMS customer-managed encryption key
    (CMEK) is preserved for transfers between Google Cloud Storage buckets. If
    unspecified, the default behavior is the same as
    KMS_KEY_DESTINATION_BUCKET_DEFAULT.

    Values:
      KMS_KEY_UNSPECIFIED: KmsKey behavior is unspecified.
      KMS_KEY_DESTINATION_BUCKET_DEFAULT: Use the destination bucket's default
        encryption settings.
      KMS_KEY_PRESERVE: Preserve the object's original Cloud KMS customer-
        managed encryption key (CMEK) if present. Objects that do not use a
        Cloud KMS encryption key will be encrypted using the destination
        bucket's encryption settings.
    """
        KMS_KEY_UNSPECIFIED = 0
        KMS_KEY_DESTINATION_BUCKET_DEFAULT = 1
        KMS_KEY_PRESERVE = 2

    class ModeValueValuesEnum(_messages.Enum):
        """Specifies how each file's mode attribute should be handled by the
    transfer. By default, mode is not preserved. Only applicable to transfers
    involving POSIX file systems, and ignored for other transfers.

    Values:
      MODE_UNSPECIFIED: Mode behavior is unspecified.
      MODE_SKIP: Do not preserve mode during a transfer job.
      MODE_PRESERVE: Preserve mode during a transfer job.
    """
        MODE_UNSPECIFIED = 0
        MODE_SKIP = 1
        MODE_PRESERVE = 2

    class StorageClassValueValuesEnum(_messages.Enum):
        """Specifies the storage class to set on objects being transferred to
    Google Cloud Storage buckets. If unspecified, the default behavior is the
    same as STORAGE_CLASS_DESTINATION_BUCKET_DEFAULT.

    Values:
      STORAGE_CLASS_UNSPECIFIED: Storage class behavior is unspecified.
      STORAGE_CLASS_DESTINATION_BUCKET_DEFAULT: Use the destination bucket's
        default storage class.
      STORAGE_CLASS_PRESERVE: Preserve the object's original storage class.
        This is only supported for transfers from Google Cloud Storage
        buckets. REGIONAL and MULTI_REGIONAL storage classes will be mapped to
        STANDARD to ensure they can be written to the destination bucket.
      STORAGE_CLASS_STANDARD: Set the storage class to STANDARD.
      STORAGE_CLASS_NEARLINE: Set the storage class to NEARLINE.
      STORAGE_CLASS_COLDLINE: Set the storage class to COLDLINE.
      STORAGE_CLASS_ARCHIVE: Set the storage class to ARCHIVE.
    """
        STORAGE_CLASS_UNSPECIFIED = 0
        STORAGE_CLASS_DESTINATION_BUCKET_DEFAULT = 1
        STORAGE_CLASS_PRESERVE = 2
        STORAGE_CLASS_STANDARD = 3
        STORAGE_CLASS_NEARLINE = 4
        STORAGE_CLASS_COLDLINE = 5
        STORAGE_CLASS_ARCHIVE = 6

    class SymlinkValueValuesEnum(_messages.Enum):
        """Specifies how symlinks should be handled by the transfer. By default,
    symlinks are not preserved. Only applicable to transfers involving POSIX
    file systems, and ignored for other transfers.

    Values:
      SYMLINK_UNSPECIFIED: Symlink behavior is unspecified.
      SYMLINK_SKIP: Do not preserve symlinks during a transfer job.
      SYMLINK_PRESERVE: Preserve symlinks during a transfer job.
    """
        SYMLINK_UNSPECIFIED = 0
        SYMLINK_SKIP = 1
        SYMLINK_PRESERVE = 2

    class TemporaryHoldValueValuesEnum(_messages.Enum):
        """Specifies how each object's temporary hold status should be preserved
    for transfers between Google Cloud Storage buckets. If unspecified, the
    default behavior is the same as TEMPORARY_HOLD_PRESERVE.

    Values:
      TEMPORARY_HOLD_UNSPECIFIED: Temporary hold behavior is unspecified.
      TEMPORARY_HOLD_SKIP: Do not set a temporary hold on the destination
        object.
      TEMPORARY_HOLD_PRESERVE: Preserve the object's original temporary hold
        status.
    """
        TEMPORARY_HOLD_UNSPECIFIED = 0
        TEMPORARY_HOLD_SKIP = 1
        TEMPORARY_HOLD_PRESERVE = 2

    class TimeCreatedValueValuesEnum(_messages.Enum):
        """Specifies how each object's `timeCreated` metadata is preserved for
    transfers. If unspecified, the default behavior is the same as
    TIME_CREATED_SKIP. This behavior is supported for transfers to GCS buckets
    from GCS, S3, Azure, S3 Compatible, and Azure sources.

    Values:
      TIME_CREATED_UNSPECIFIED: TimeCreated behavior is unspecified.
      TIME_CREATED_SKIP: Do not preserve the `timeCreated` metadata from the
        source object.
      TIME_CREATED_PRESERVE_AS_CUSTOM_TIME: Preserves the source object's
        `timeCreated` or `lastModified` metadata in the `customTime` field in
        the destination object. Note that any value stored in the source
        object's `customTime` field will not be propagated to the destination
        object.
    """
        TIME_CREATED_UNSPECIFIED = 0
        TIME_CREATED_SKIP = 1
        TIME_CREATED_PRESERVE_AS_CUSTOM_TIME = 2

    class UidValueValuesEnum(_messages.Enum):
        """Specifies how each file's POSIX user ID (UID) attribute should be
    handled by the transfer. By default, UID is not preserved. Only applicable
    to transfers involving POSIX file systems, and ignored for other
    transfers.

    Values:
      UID_UNSPECIFIED: UID behavior is unspecified.
      UID_SKIP: Do not preserve UID during a transfer job.
      UID_NUMBER: Preserve UID during a transfer job.
    """
        UID_UNSPECIFIED = 0
        UID_SKIP = 1
        UID_NUMBER = 2
    acl = _messages.EnumField('AclValueValuesEnum', 1)
    gid = _messages.EnumField('GidValueValuesEnum', 2)
    kmsKey = _messages.EnumField('KmsKeyValueValuesEnum', 3)
    mode = _messages.EnumField('ModeValueValuesEnum', 4)
    storageClass = _messages.EnumField('StorageClassValueValuesEnum', 5)
    symlink = _messages.EnumField('SymlinkValueValuesEnum', 6)
    temporaryHold = _messages.EnumField('TemporaryHoldValueValuesEnum', 7)
    timeCreated = _messages.EnumField('TimeCreatedValueValuesEnum', 8)
    uid = _messages.EnumField('UidValueValuesEnum', 9)