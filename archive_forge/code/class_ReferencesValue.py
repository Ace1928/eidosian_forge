from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ReferencesValue(_messages.Message):
    """Required.

    Messages:
      AdditionalProperty: An additional property for a ReferencesValue object.

    Fields:
      additionalProperties: Additional properties of type ReferencesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ReferencesValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1PublisherModelResourceReference
          attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelResourceReference', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)