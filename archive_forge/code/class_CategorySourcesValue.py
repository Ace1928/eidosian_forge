from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CategorySourcesValue(_messages.Message):
    """Map of enabled categories as keys and the policy that enabled it as
    values. For example, the key can be `categories/google` and value can be
    `{"projects/123/consumerPolicies/default",
    "folders/456/consumerPolicies/default"}` where the category is enabled and
    the order of the resource list is nearest first in the hierarchy.

    Messages:
      AdditionalProperty: An additional property for a CategorySourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type CategorySourcesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CategorySourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyList attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PolicyList', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)