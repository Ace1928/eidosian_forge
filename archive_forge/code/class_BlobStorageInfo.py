from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlobStorageInfo(_messages.Message):
    """BlobStorageInfo contains details about the data stored in Blob Storage
  for the referenced resource. Note: Storage class is only valid for DICOM and
  hence will only be populated for DICOM resources.

  Enums:
    StorageClassValueValuesEnum: The storage class in which the Blob data is
      stored.

  Fields:
    sizeBytes: Size in bytes of data stored in Blob Storage.
    storageClass: The storage class in which the Blob data is stored.
    storageClassUpdateTime: The time at which the storage class was updated.
      This is used to compute early deletion fees of the resource.
  """

    class StorageClassValueValuesEnum(_messages.Enum):
        """The storage class in which the Blob data is stored.

    Values:
      BLOB_STORAGE_CLASS_UNSPECIFIED: If unspecified in CreateDataset, the
        StorageClass defaults to STANDARD. If unspecified in UpdateDataset and
        the StorageClass is set in the field mask, an InvalidRequest error is
        thrown.
      STANDARD: This stores the Object in Blob Standard Storage:
        https://cloud.google.com/storage/docs/storage-classes#standard
      NEARLINE: This stores the Object in Blob Nearline Storage:
        https://cloud.google.com/storage/docs/storage-classes#nearline
      COLDLINE: This stores the Object in Blob Coldline Storage:
        https://cloud.google.com/storage/docs/storage-classes#coldline
      ARCHIVE: This stores the Object in Blob Archive Storage:
        https://cloud.google.com/storage/docs/storage-classes#archive
    """
        BLOB_STORAGE_CLASS_UNSPECIFIED = 0
        STANDARD = 1
        NEARLINE = 2
        COLDLINE = 3
        ARCHIVE = 4
    sizeBytes = _messages.IntegerField(1)
    storageClass = _messages.EnumField('StorageClassValueValuesEnum', 2)
    storageClassUpdateTime = _messages.StringField(3)