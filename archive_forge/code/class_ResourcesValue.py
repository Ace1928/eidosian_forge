from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ResourcesValue(_messages.Message):
    """Resources referenceable within a workflow.

    Messages:
      AdditionalProperty: An additional property for a ResourcesValue object.

    Fields:
      additionalProperties: Additional properties of type ResourcesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A Resource attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Resource', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)