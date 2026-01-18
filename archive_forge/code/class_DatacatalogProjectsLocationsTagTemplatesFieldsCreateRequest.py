from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesFieldsCreateRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesFieldsCreateRequest object.

  Fields:
    googleCloudDatacatalogV1TagTemplateField: A
      GoogleCloudDatacatalogV1TagTemplateField resource to be passed as the
      request body.
    parent: Required. The name of the project and the template location
      [region](https://cloud.google.com/data-catalog/docs/concepts/regions).
    tagTemplateFieldId: Required. The ID of the tag template field to create.
      Note: Adding a required field to an existing template is *not* allowed.
      Field IDs can contain letters (both uppercase and lowercase), numbers
      (0-9), underscores (_) and dashes (-). Field IDs must be at least 1
      character long and at most 128 characters long. Field IDs must also be
      unique within their template.
  """
    googleCloudDatacatalogV1TagTemplateField = _messages.MessageField('GoogleCloudDatacatalogV1TagTemplateField', 1)
    parent = _messages.StringField(2, required=True)
    tagTemplateFieldId = _messages.StringField(3)