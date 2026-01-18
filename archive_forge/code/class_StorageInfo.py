from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageInfo(_messages.Message):
    """StorageInfo encapsulates all the storage info of a resource.

  Fields:
    blobStorageInfo: Info about the data stored in blob storage for the
      resource.
    referencedResource: The resource whose storage info is returned. For
      example, to specify the resource path of a DICOM Instance: `projects/{pr
      ojectID}/locations/{locationID}/datasets/{datasetID}/dicomStores/{dicom_
      store_id}/dicomWeb/studi/{study_uid}/series/{series_uid}/instances/{inst
      ance_uid}`
    structuredStorageInfo: Info about the data stored in structured storage
      for the resource.
  """
    blobStorageInfo = _messages.MessageField('BlobStorageInfo', 1)
    referencedResource = _messages.StringField(2)
    structuredStorageInfo = _messages.MessageField('StructuredStorageInfo', 3)