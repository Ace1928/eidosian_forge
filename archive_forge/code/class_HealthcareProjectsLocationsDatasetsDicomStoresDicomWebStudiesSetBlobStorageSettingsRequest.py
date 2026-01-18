from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSetBlobStorageSettingsRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSetBlobSt
  orageSettingsRequest object.

  Fields:
    resource: Required. The path of the resource to update the blob storage
      settings in the format of `projects/{projectID}/locations/{locationID}/d
      atasets/{datasetID}/dicomStores/{dicomStoreID}/dicomWeb/studies/{studyUI
      D}`, `projects/{projectID}/locations/{locationID}/datasets/{datasetID}/d
      icomStores/{dicomStoreID}/dicomWeb/studies/{studyUID}/series/{seriesUID}
      /`, or `projects/{projectID}/locations/{locationID}/datasets/{datasetID}
      /dicomStores/{dicomStoreID}/dicomWeb/studies/{studyUID}/series/{seriesUI
      D}/instances/{instanceUID}`. If `filter_config` is specified, set the
      value of `resource` to the resource name of a DICOM store in the format
      `projects/{projectID}/locations/{locationID}/datasets/{datasetID}/dicomS
      tores/{dicomStoreID}`.
    setBlobStorageSettingsRequest: A SetBlobStorageSettingsRequest resource to
      be passed as the request body.
  """
    resource = _messages.StringField(1, required=True)
    setBlobStorageSettingsRequest = _messages.MessageField('SetBlobStorageSettingsRequest', 2)