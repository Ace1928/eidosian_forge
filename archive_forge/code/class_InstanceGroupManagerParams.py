from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerParams(_messages.Message):
    """Input only additional params for instance group manager creation.

  Messages:
    ResourceManagerTagsValue: Resource manager tags to bind to the managed
      instance group. The tags are key-value pairs. Keys must be in the format
      tagKeys/123 and values in the format tagValues/456. For more
      information, see Manage tags for resources.

  Fields:
    resourceManagerTags: Resource manager tags to bind to the managed instance
      group. The tags are key-value pairs. Keys must be in the format
      tagKeys/123 and values in the format tagValues/456. For more
      information, see Manage tags for resources.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceManagerTagsValue(_messages.Message):
        """Resource manager tags to bind to the managed instance group. The tags
    are key-value pairs. Keys must be in the format tagKeys/123 and values in
    the format tagValues/456. For more information, see Manage tags for
    resources.

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