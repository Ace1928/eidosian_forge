from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ResourceTagsValue(_messages.Message):
    """[Optional] The tags associated with this table. Tag keys are globally
    unique. See additional information on
    [tags](https://cloud.google.com/iam/docs/tags-access-control#definitions).
    An object containing a list of "key": value pairs. The key is the
    namespaced friendly name of the tag key, e.g. "12345/environment" where
    12345 is parent id. The value is the friendly short name of the tag value,
    e.g. "production".

    Messages:
      AdditionalProperty: An additional property for a ResourceTagsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ResourceTagsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ResourceTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)