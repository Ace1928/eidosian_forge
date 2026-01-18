from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdminConsents(_messages.Message):
    """List of admin Consent resources to be applied.

  Fields:
    names: The versioned names of the admin Consent resource(s), in the format
      `projects/{project_id}/locations/{location}/datasets/{dataset_id}/fhirSt
      ores/{fhir_store_id}/fhir/Consent/{resource_id}/_history/{version_id}`.
      For FHIR stores with `disable_resource_versioning=true`, the format is `
      projects/{project_id}/locations/{location}/datasets/{dataset_id}/fhirSto
      res/{fhir_store_id}/fhir/Consent/{resource_id}`.
  """
    names = _messages.StringField(1, repeated=True)