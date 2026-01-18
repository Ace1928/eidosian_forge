from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsPa
  tchRequest object.

  Fields:
    attributeDefinition: A AttributeDefinition resource to be passed as the
      request body.
    name: Identifier. Resource name of the Attribute definition, of the form `
      projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/cons
      entStores/{consent_store_id}/attributeDefinitions/{attribute_definition_
      id}`. Cannot be changed after creation.
    updateMask: Required. The update mask that applies to the resource. For
      the `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask. Only the
      `description`, `allowed_values`, `consent_default_values` and
      `data_mapping_default_value` fields can be updated. The updated
      `allowed_values` must contain all values from the previous
      `allowed_values`.
  """
    attributeDefinition = _messages.MessageField('AttributeDefinition', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)