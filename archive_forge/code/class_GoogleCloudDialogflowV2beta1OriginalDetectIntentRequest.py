from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1OriginalDetectIntentRequest(_messages.Message):
    """Represents the contents of the original request that was passed to the
  `[Streaming]DetectIntent` call.

  Messages:
    PayloadValue: Optional. This field is set to the value of the
      `QueryParameters.payload` field passed in the request. Some integrations
      that query a Dialogflow agent may provide additional information in the
      payload. In particular, for the Dialogflow Phone Gateway integration,
      this field has the form: { "telephony": { "caller_id": "+18558363987" }
      } Note: The caller ID field (`caller_id`) will be redacted for Trial
      Edition agents and populated with the caller ID in [E.164
      format](https://en.wikipedia.org/wiki/E.164) for Essentials Edition
      agents.

  Fields:
    payload: Optional. This field is set to the value of the
      `QueryParameters.payload` field passed in the request. Some integrations
      that query a Dialogflow agent may provide additional information in the
      payload. In particular, for the Dialogflow Phone Gateway integration,
      this field has the form: { "telephony": { "caller_id": "+18558363987" }
      } Note: The caller ID field (`caller_id`) will be redacted for Trial
      Edition agents and populated with the caller ID in [E.164
      format](https://en.wikipedia.org/wiki/E.164) for Essentials Edition
      agents.
    source: The source of this request, e.g., `google`, `facebook`, `slack`.
      It is set by Dialogflow-owned servers.
    version: Optional. The version of the protocol used for this request. This
      field is AoG-specific.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """Optional. This field is set to the value of the
    `QueryParameters.payload` field passed in the request. Some integrations
    that query a Dialogflow agent may provide additional information in the
    payload. In particular, for the Dialogflow Phone Gateway integration, this
    field has the form: { "telephony": { "caller_id": "+18558363987" } } Note:
    The caller ID field (`caller_id`) will be redacted for Trial Edition
    agents and populated with the caller ID in [E.164
    format](https://en.wikipedia.org/wiki/E.164) for Essentials Edition
    agents.

    Messages:
      AdditionalProperty: An additional property for a PayloadValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    payload = _messages.MessageField('PayloadValue', 1)
    source = _messages.StringField(2)
    version = _messages.StringField(3)