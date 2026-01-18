from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsCr
  eateRequest object.

  Fields:
    attributeDefinition: A AttributeDefinition resource to be passed as the
      request body.
    attributeDefinitionId: Required. The ID of the Attribute definition to
      create. The string must match the following regex: `_a-zA-Z{0,255}` and
      must not be a reserved keyword within the Common Expression Language as
      listed on https://github.com/google/cel-spec/blob/master/doc/langdef.md.
    parent: Required. The name of the consent store that this Attribute
      definition belongs to.
  """
    attributeDefinition = _messages.MessageField('AttributeDefinition', 1)
    attributeDefinitionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)