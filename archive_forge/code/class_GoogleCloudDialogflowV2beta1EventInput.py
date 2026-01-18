from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1EventInput(_messages.Message):
    """Events allow for matching intents by event name instead of the natural
  language input. For instance, input `` can trigger a personalized welcome
  response. The parameter `name` may be used by the agent in the response:
  `"Hello #welcome_event.name! What can I do for you today?"`.

  Messages:
    ParametersValue: The collection of parameters associated with the event.
      Depending on your protocol or client library language, this is a map,
      associative array, symbol table, dictionary, or JSON object composed of
      a collection of (MapKey, MapValue) pairs: * MapKey type: string * MapKey
      value: parameter name * MapValue type: If parameter's entity type is a
      composite entity then use map, otherwise, depending on the parameter
      value type, it could be one of string, number, boolean, null, list or
      map. * MapValue value: If parameter's entity type is a composite entity
      then use map from composite entity property names to property values,
      otherwise, use parameter value.

  Fields:
    languageCode: Required. The language of this query. See [Language
      Support](https://cloud.google.com/dialogflow/docs/reference/language)
      for a list of the currently supported language codes. Note that queries
      in the same session do not necessarily need to specify the same
      language. This field is ignored when used in the context of a
      WebhookResponse.followup_event_input field, because the language was
      already defined in the originating detect intent request.
    name: Required. The unique identifier of the event.
    parameters: The collection of parameters associated with the event.
      Depending on your protocol or client library language, this is a map,
      associative array, symbol table, dictionary, or JSON object composed of
      a collection of (MapKey, MapValue) pairs: * MapKey type: string * MapKey
      value: parameter name * MapValue type: If parameter's entity type is a
      composite entity then use map, otherwise, depending on the parameter
      value type, it could be one of string, number, boolean, null, list or
      map. * MapValue value: If parameter's entity type is a composite entity
      then use map from composite entity property names to property values,
      otherwise, use parameter value.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """The collection of parameters associated with the event. Depending on
    your protocol or client library language, this is a map, associative
    array, symbol table, dictionary, or JSON object composed of a collection
    of (MapKey, MapValue) pairs: * MapKey type: string * MapKey value:
    parameter name * MapValue type: If parameter's entity type is a composite
    entity then use map, otherwise, depending on the parameter value type, it
    could be one of string, number, boolean, null, list or map. * MapValue
    value: If parameter's entity type is a composite entity then use map from
    composite entity property names to property values, otherwise, use
    parameter value.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    languageCode = _messages.StringField(1)
    name = _messages.StringField(2)
    parameters = _messages.MessageField('ParametersValue', 3)