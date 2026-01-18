from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesFieldsEnumValuesRenameRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesFieldsEnumValuesRenameRequest
  object.

  Fields:
    googleCloudDatacatalogV1RenameTagTemplateFieldEnumValueRequest: A
      GoogleCloudDatacatalogV1RenameTagTemplateFieldEnumValueRequest resource
      to be passed as the request body.
    name: Required. The name of the enum field value.
  """
    googleCloudDatacatalogV1RenameTagTemplateFieldEnumValueRequest = _messages.MessageField('GoogleCloudDatacatalogV1RenameTagTemplateFieldEnumValueRequest', 1)
    name = _messages.StringField(2, required=True)