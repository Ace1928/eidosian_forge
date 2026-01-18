from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesFieldsPatchRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesFieldsPatchRequest object.

  Fields:
    googleCloudDatacatalogV1TagTemplateField: A
      GoogleCloudDatacatalogV1TagTemplateField resource to be passed as the
      request body.
    name: Required. The name of the tag template field.
    updateMask: Optional. Names of fields whose values to overwrite on an
      individual field of a tag template. The following fields are modifiable:
      * `display_name` * `type.enum_type` * `is_required` If this parameter is
      absent or empty, all modifiable fields are overwritten. If such fields
      are non-required and omitted in the request body, their values are
      emptied with one exception: when updating an enum type, the provided
      values are merged with the existing values. Therefore, enum values can
      only be added, existing enum values cannot be deleted or renamed.
      Additionally, updating a template field from optional to required is
      *not* allowed.
  """
    googleCloudDatacatalogV1TagTemplateField = _messages.MessageField('GoogleCloudDatacatalogV1TagTemplateField', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)