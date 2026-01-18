from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskParams(_messages.Message):
    """Additional disk params.

  Messages:
    ResourceManagerTagsValue: Resource manager tags to be bound to the disk.
      Tag keys and values have the same definition as resource manager tags.
      Keys must be in the format `tagKeys/{tag_key_id}`, and values are in the
      format `tagValues/456`. The field is ignored (both PUT & PATCH) when
      empty.

  Fields:
    resourceManagerTags: Resource manager tags to be bound to the disk. Tag
      keys and values have the same definition as resource manager tags. Keys
      must be in the format `tagKeys/{tag_key_id}`, and values are in the
      format `tagValues/456`. The field is ignored (both PUT & PATCH) when
      empty.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceManagerTagsValue(_messages.Message):
        """Resource manager tags to be bound to the disk. Tag keys and values
    have the same definition as resource manager tags. Keys must be in the
    format `tagKeys/{tag_key_id}`, and values are in the format
    `tagValues/456`. The field is ignored (both PUT & PATCH) when empty.

    Messages:
      AdditionalProperty: An additional property for a
        ResourceManagerTagsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ResourceManagerTagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceManagerTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    resourceManagerTags = _messages.MessageField('ResourceManagerTagsValue', 1)