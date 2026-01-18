from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesFieldsRenameRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesFieldsRenameRequest object.

  Fields:
    googleCloudDatacatalogV1RenameTagTemplateFieldRequest: A
      GoogleCloudDatacatalogV1RenameTagTemplateFieldRequest resource to be
      passed as the request body.
    name: Required. The name of the tag template field.
  """
    googleCloudDatacatalogV1RenameTagTemplateFieldRequest = _messages.MessageField('GoogleCloudDatacatalogV1RenameTagTemplateFieldRequest', 1)
    name = _messages.StringField(2, required=True)