from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class WebhookHeadersValue(_messages.Message):
    """This field can be used to pass HTTP headers for a webhook call. These
    headers will be sent to webhook along with the headers that have been
    configured through the Dialogflow web console. The headers defined within
    this field will overwrite the headers configured through the Dialogflow
    console if there is a conflict. Header names are case-insensitive.
    Google's specified headers are not allowed. Including: "Host", "Content-
    Length", "Connection", "From", "User-Agent", "Accept-Encoding", "If-
    Modified-Since", "If-None-Match", "X-Forwarded-For", etc.

    Messages:
      AdditionalProperty: An additional property for a WebhookHeadersValue
        object.

    Fields:
      additionalProperties: Additional properties of type WebhookHeadersValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a WebhookHeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)