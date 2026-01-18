from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1Tag(_messages.Message):
    """Tags are used to attach custom metadata to Data Catalog resources. Tags
  conform to the specifications within their tag template. See [Data Catalog
  IAM](https://cloud.google.com/data-catalog/docs/concepts/iam) for
  information on the permissions needed to create or view tags.

  Messages:
    FieldsValue: Required. This maps the ID of a tag field to the value of and
      additional information about that field. Valid field IDs are defined by
      the tag's template. A tag must have at least 1 field and at most 500
      fields.

  Fields:
    column: Resources like Entry can have schemas associated with them. This
      scope allows users to attach tags to an individual column based on that
      schema. For attaching a tag to a nested column, use `.` to separate the
      column names. Example: * `outer_column.inner_column`
    fields: Required. This maps the ID of a tag field to the value of and
      additional information about that field. Valid field IDs are defined by
      the tag's template. A tag must have at least 1 field and at most 500
      fields.
    name: Identifier. The resource name of the tag in URL format. Example: * p
      rojects/{project_id}/locations/{location}/entrygroups/{entry_group_id}/e
      ntries/{entry_id}/tags/{tag_id} where `tag_id` is a system-generated
      identifier. Note that this Tag may not actually be stored in the
      location in this name.
    template: Required. The resource name of the tag template that this tag
      uses. Example: * projects/{project_id}/locations/{location}/tagTemplates
      /{tag_template_id} This field cannot be modified after creation.
    templateDisplayName: Output only. The display name of the tag template.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FieldsValue(_messages.Message):
        """Required. This maps the ID of a tag field to the value of and
    additional information about that field. Valid field IDs are defined by
    the tag's template. A tag must have at least 1 field and at most 500
    fields.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1beta1TagField attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1beta1TagField', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    column = _messages.StringField(1)
    fields = _messages.MessageField('FieldsValue', 2)
    name = _messages.StringField(3)
    template = _messages.StringField(4)
    templateDisplayName = _messages.StringField(5)