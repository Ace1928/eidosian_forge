from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesCreateRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesCreateRequest object.

  Fields:
    googleCloudDatacatalogV1TagTemplate: A GoogleCloudDatacatalogV1TagTemplate
      resource to be passed as the request body.
    parent: Required. The name of the project and the template location
      [region](https://cloud.google.com/data-catalog/docs/concepts/regions).
    tagTemplateId: Required. The ID of the tag template to create. The ID must
      contain only lowercase letters (a-z), numbers (0-9), or underscores (_),
      and must start with a letter or underscore. The maximum size is 64 bytes
      when encoded in UTF-8.
  """
    googleCloudDatacatalogV1TagTemplate = _messages.MessageField('GoogleCloudDatacatalogV1TagTemplate', 1)
    parent = _messages.StringField(2, required=True)
    tagTemplateId = _messages.StringField(3)