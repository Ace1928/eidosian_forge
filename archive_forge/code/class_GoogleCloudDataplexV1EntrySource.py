from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntrySource(_messages.Message):
    """EntrySource contains source system related information for the entry.

  Messages:
    LabelsValue: User-defined labels. The maximum size of keys and values is
      128 characters each.

  Fields:
    ancestors: Immutable. The ancestors of the Entry in the source system.
    createTime: The create time of the resource in the source system.
    description: Description of the Entry. The maximum size of the field is
      2000 characters.
    displayName: User friendly display name. The maximum size of the field is
      500 characters.
    labels: User-defined labels. The maximum size of keys and values is 128
      characters each.
    platform: The platform containing the source system. The maximum size of
      the field is 64 characters.
    resource: The name of the resource in the source system. The maximum size
      of the field is 4000 characters.
    system: The name of the source system. The maximum size of the field is 64
      characters.
    updateTime: The update time of the resource in the source system.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-defined labels. The maximum size of keys and values is 128
    characters each.

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
    ancestors = _messages.MessageField('GoogleCloudDataplexV1EntrySourceAncestor', 1, repeated=True)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    platform = _messages.StringField(6)
    resource = _messages.StringField(7)
    system = _messages.StringField(8)
    updateTime = _messages.StringField(9)