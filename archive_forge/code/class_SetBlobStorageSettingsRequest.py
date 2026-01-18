from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetBlobStorageSettingsRequest(_messages.Message):
    """Request message for `SetBlobStorageSettings` method.

  Fields:
    blobStorageSettings: The blob storage settings to update for the specified
      resources. Only fields listed in `update_mask` are applied.
    filterConfig: Optional. A filter configuration. If `filter_config` is
      specified, set the value of `resource` to the resource name of a DICOM
      store in the format `projects/{projectID}/locations/{locationID}/dataset
      s/{datasetID}/dicomStores/{dicomStoreID}`.
  """
    blobStorageSettings = _messages.MessageField('BlobStorageSettings', 1)
    filterConfig = _messages.MessageField('DicomFilterConfig', 2)