from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresPatchRequest object.

  Fields:
    consentStore: A ConsentStore resource to be passed as the request body.
    name: Identifier. Resource name of the consent store, of the form `project
      s/{project_id}/locations/{location_id}/datasets/{dataset_id}/consentStor
      es/{consent_store_id}`. Cannot be changed after creation.
    updateMask: Required. The update mask that applies to the resource. For
      the `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask. Only the `labels`,
      `default_consent_ttl`, and `enable_consent_create_on_update` fields are
      allowed to be updated.
  """
    consentStore = _messages.MessageField('ConsentStore', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)