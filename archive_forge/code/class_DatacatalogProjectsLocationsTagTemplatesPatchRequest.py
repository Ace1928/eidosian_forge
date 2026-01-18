from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesPatchRequest object.

  Fields:
    googleCloudDatacatalogV1TagTemplate: A GoogleCloudDatacatalogV1TagTemplate
      resource to be passed as the request body.
    name: Identifier. The resource name of the tag template in URL format.
      Note: The tag template itself and its child resources might not be
      stored in the location specified in its name.
    updateMask: Names of fields whose values to overwrite on a tag template.
      Currently, only `display_name` and `is_publicly_readable` can be
      overwritten. If this parameter is absent or empty, all modifiable fields
      are overwritten. If such fields are non-required and omitted in the
      request body, their values are emptied. Note: Updating the
      `is_publicly_readable` field may require up to 12 hours to take effect
      in search results.
  """
    googleCloudDatacatalogV1TagTemplate = _messages.MessageField('GoogleCloudDatacatalogV1TagTemplate', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)