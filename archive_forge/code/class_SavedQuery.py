from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SavedQuery(_messages.Message):
    """A saved query which can be shared with others or used later.

  Messages:
    LabelsValue: Labels applied on the resource. This value should not contain
      more than 10 entries. The key and value of each entry must be non-empty
      and fewer than 64 characters.

  Fields:
    content: The query content.
    createTime: Output only. The create time of this saved query.
    creator: Output only. The account's email address who has created this
      saved query.
    description: The description of this saved query. This value should be
      fewer than 255 characters.
    labels: Labels applied on the resource. This value should not contain more
      than 10 entries. The key and value of each entry must be non-empty and
      fewer than 64 characters.
    lastUpdateTime: Output only. The last update time of this saved query.
    lastUpdater: Output only. The account's email address who has updated this
      saved query most recently.
    name: The resource name of the saved query. The format must be: *
      projects/project_number/savedQueries/saved_query_id *
      folders/folder_number/savedQueries/saved_query_id *
      organizations/organization_number/savedQueries/saved_query_id
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels applied on the resource. This value should not contain more
    than 10 entries. The key and value of each entry must be non-empty and
    fewer than 64 characters.

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
    content = _messages.MessageField('QueryContent', 1)
    createTime = _messages.StringField(2)
    creator = _messages.StringField(3)
    description = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    lastUpdateTime = _messages.StringField(6)
    lastUpdater = _messages.StringField(7)
    name = _messages.StringField(8)