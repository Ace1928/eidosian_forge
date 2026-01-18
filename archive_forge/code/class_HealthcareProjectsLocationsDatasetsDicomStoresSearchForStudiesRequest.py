from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresSearchForStudiesRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresSearchForStudiesRequest
  object.

  Fields:
    dicomWebPath: The path of the SearchForStudies DICOMweb request. For
      example, `studies`.
    parent: The name of the DICOM store that is being accessed. For example, `
      projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/dico
      mStores/{dicom_store_id}`.
  """
    dicomWebPath = _messages.StringField(1, required=True)
    parent = _messages.StringField(2, required=True)