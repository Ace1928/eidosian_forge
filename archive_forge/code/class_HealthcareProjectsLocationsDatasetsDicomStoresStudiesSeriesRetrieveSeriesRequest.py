from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesRetrieveSeriesRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresStudiesSeriesRetrieveSer
  iesRequest object.

  Fields:
    dicomWebPath: The path of the RetrieveSeries DICOMweb request. For
      example, `studies/{study_uid}/series/{series_uid}`.
    parent: The name of the DICOM store that is being accessed. For example, `
      projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/dico
      mStores/{dicom_store_id}`.
  """
    dicomWebPath = _messages.StringField(1, required=True)
    parent = _messages.StringField(2, required=True)