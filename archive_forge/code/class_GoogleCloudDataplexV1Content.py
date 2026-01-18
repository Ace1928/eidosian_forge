from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Content(_messages.Message):
    """Content represents a user-visible notebook or a sql script

  Messages:
    LabelsValue: Optional. User defined labels for the content.

  Fields:
    createTime: Output only. Content creation time.
    dataText: Required. Content data in string format.
    description: Optional. Description of the content.
    labels: Optional. User defined labels for the content.
    name: Output only. The relative resource name of the content, of the form:
      projects/{project_id}/locations/{location_id}/lakes/{lake_id}/content/{c
      ontent_id}
    notebook: Notebook related configurations.
    path: Required. The path for the Content file, represented as directory
      structure. Unique within a lake. Limited to alphanumerics, hyphens,
      underscores, dots and slashes.
    sqlScript: Sql Script related configurations.
    uid: Output only. System generated globally unique ID for the content.
      This ID will be different if the content is deleted and re-created with
      the same name.
    updateTime: Output only. The time when the content was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User defined labels for the content.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    dataText = _messages.StringField(2)
    description = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    notebook = _messages.MessageField('GoogleCloudDataplexV1ContentNotebook', 6)
    path = _messages.StringField(7)
    sqlScript = _messages.MessageField('GoogleCloudDataplexV1ContentSqlScript', 8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)