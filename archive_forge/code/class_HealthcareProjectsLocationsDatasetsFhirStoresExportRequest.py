from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresExportRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresExportRequest object.

  Fields:
    exportResourcesRequest: A ExportResourcesRequest resource to be passed as
      the request body.
    name: Required. The name of the FHIR store to export resource from, in the
      format `projects/{project_id}/locations/{location_id}/datasets/{dataset_
      id}/fhirStores/{fhir_store_id}`.
  """
    exportResourcesRequest = _messages.MessageField('ExportResourcesRequest', 1)
    name = _messages.StringField(2, required=True)