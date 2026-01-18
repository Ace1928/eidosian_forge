from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsPatchRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsPatchRequest
  object.

  Fields:
    name: Resource name of the User data mapping, of the form `projects/{proje
      ct_id}/locations/{location_id}/datasets/{dataset_id}/consentStores/{cons
      ent_store_id}/userDataMappings/{user_data_mapping_id}`.
    updateMask: Required. The update mask that applies to the resource. For
      the `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask. Only the `data_id`,
      `user_id` and `resource_attributes` fields can be updated.
    userDataMapping: A UserDataMapping resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    updateMask = _messages.StringField(2)
    userDataMapping = _messages.MessageField('UserDataMapping', 3)