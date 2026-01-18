from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessage(_messages.Message):
    """A rich response message. Corresponds to the intent `Response` field in
  the Dialogflow console. For more information, see [Rich response
  messages](https://cloud.google.com/dialogflow/docs/intents-rich-messages).

  Enums:
    PlatformValueValuesEnum: Optional. The platform that this message is
      intended for.

  Messages:
    PayloadValue: A custom platform-specific response.

  Fields:
    basicCard: The basic card response for Actions on Google.
    browseCarouselCard: Browse carousel card for Actions on Google.
    card: The card response.
    carouselSelect: The carousel card response for Actions on Google.
    image: The image response.
    linkOutSuggestion: The link out suggestion chip for Actions on Google.
    listSelect: The list card response for Actions on Google.
    mediaContent: The media content card for Actions on Google.
    payload: A custom platform-specific response.
    platform: Optional. The platform that this message is intended for.
    quickReplies: The quick replies response.
    simpleResponses: The voice and text-only responses for Actions on Google.
    suggestions: The suggestion chips for Actions on Google.
    tableCard: Table card for Actions on Google.
    text: The text response.
  """

    class PlatformValueValuesEnum(_messages.Enum):
        """Optional. The platform that this message is intended for.

    Values:
      PLATFORM_UNSPECIFIED: Default platform.
      FACEBOOK: Facebook.
      SLACK: Slack.
      TELEGRAM: Telegram.
      KIK: Kik.
      SKYPE: Skype.
      LINE: Line.
      VIBER: Viber.
      ACTIONS_ON_GOOGLE: Google Assistant See [Dialogflow webhook format](http
        s://developers.google.com/assistant/actions/build/json/dialogflow-
        webhook-json)
      GOOGLE_HANGOUTS: Google Hangouts.
    """
        PLATFORM_UNSPECIFIED = 0
        FACEBOOK = 1
        SLACK = 2
        TELEGRAM = 3
        KIK = 4
        SKYPE = 5
        LINE = 6
        VIBER = 7
        ACTIONS_ON_GOOGLE = 8
        GOOGLE_HANGOUTS = 9

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """A custom platform-specific response.

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
    basicCard = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageBasicCard', 1)
    browseCarouselCard = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageBrowseCarouselCard', 2)
    card = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageCard', 3)
    carouselSelect = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageCarouselSelect', 4)
    image = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageImage', 5)
    linkOutSuggestion = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageLinkOutSuggestion', 6)
    listSelect = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageListSelect', 7)
    mediaContent = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageMediaContent', 8)
    payload = _messages.MessageField('PayloadValue', 9)
    platform = _messages.EnumField('PlatformValueValuesEnum', 10)
    quickReplies = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageQuickReplies', 11)
    simpleResponses = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageSimpleResponses', 12)
    suggestions = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageSuggestions', 13)
    tableCard = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageTableCard', 14)
    text = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageText', 15)