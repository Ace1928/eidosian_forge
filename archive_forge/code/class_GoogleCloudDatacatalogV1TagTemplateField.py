from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1TagTemplateField(_messages.Message):
    """The template for an individual field within a tag template.

  Fields:
    description: The description for this field. Defaults to an empty string.
    displayName: The display name for this field. Defaults to an empty string.
      The name must contain only Unicode letters, numbers (0-9), underscores
      (_), dashes (-), spaces ( ), and can't start or end with spaces. The
      maximum length is 200 characters.
    isRequired: If true, this field is required. Defaults to false.
    name: Identifier. The resource name of the tag template field in URL
      format. Example: `projects/{PROJECT_ID}/locations/{LOCATION}/tagTemplate
      s/{TAG_TEMPLATE}/fields/{FIELD}` Note: The tag template field itself
      might not be stored in the location specified in its name. The name must
      contain only letters (a-z, A-Z), numbers (0-9), or underscores (_), and
      must start with a letter or underscore. The maximum length is 64
      characters.
    order: The order of this field with respect to other fields in this tag
      template. For example, a higher value can indicate a more important
      field. The value can be negative. Multiple fields can have the same
      order and field orders within a tag don't have to be sequential.
    type: Required. The type of value this tag field can contain.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    isRequired = _messages.BooleanField(3)
    name = _messages.StringField(4)
    order = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    type = _messages.MessageField('GoogleCloudDatacatalogV1FieldType', 6)