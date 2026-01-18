from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class RequestHeadersValue(_messages.Message):
    """Optional. The HTTP request headers to send together with fulfillment
    requests.

    Messages:
      AdditionalProperty: An additional property for a RequestHeadersValue
        object.

    Fields:
      additionalProperties: Additional properties of type RequestHeadersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a RequestHeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)