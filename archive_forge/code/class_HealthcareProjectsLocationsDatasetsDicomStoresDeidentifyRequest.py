from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresDeidentifyRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresDeidentifyRequest
  object.

  Fields:
    deidentifyDicomStoreRequest: A DeidentifyDicomStoreRequest resource to be
      passed as the request body.
    sourceStore: Required. Source DICOM store resource name. For example, `pro
      jects/{project_id}/locations/{location_id}/datasets/{dataset_id}/dicomSt
      ores/{dicom_store_id}`.
  """
    deidentifyDicomStoreRequest = _messages.MessageField('DeidentifyDicomStoreRequest', 1)
    sourceStore = _messages.StringField(2, required=True)