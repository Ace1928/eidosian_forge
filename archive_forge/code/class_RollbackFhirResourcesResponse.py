from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackFhirResourcesResponse(_messages.Message):
    """Final response of rollback FHIR resources request.

  Fields:
    fhirStore: The name of the FHIR store to rollback, in the format of
      "projects/{project_id}/locations/{location_id}/datasets/{dataset_id}
      /fhirStores/{fhir_store_id}".
  """
    fhirStore = _messages.StringField(1)