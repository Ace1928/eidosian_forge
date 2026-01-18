from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ResourceManagerTagsValue(_messages.Message):
    """Optional. Resource manager tags to be bound to the instance.

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