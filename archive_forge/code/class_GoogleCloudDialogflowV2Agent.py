from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2Agent(_messages.Message):
    """A Dialogflow agent is a virtual agent that handles conversations with
  your end-users. It is a natural language understanding module that
  understands the nuances of human language. Dialogflow translates end-user
  text or audio during a conversation to structured data that your apps and
  services can understand. You design and build a Dialogflow agent to handle
  the types of conversations required for your system. For more information
  about agents, see the [Agent
  guide](https://cloud.google.com/dialogflow/docs/agents-overview).

  Enums:
    ApiVersionValueValuesEnum: Optional. API version displayed in Dialogflow
      console. If not specified, V2 API is assumed. Clients are free to query
      different service endpoints for different API versions. However, bots
      connectors and webhook calls will follow the specified API version.
    MatchModeValueValuesEnum: Optional. Determines how intents are detected
      from user queries.
    TierValueValuesEnum: Optional. The agent tier. If not specified,
      TIER_STANDARD is assumed.

  Fields:
    apiVersion: Optional. API version displayed in Dialogflow console. If not
      specified, V2 API is assumed. Clients are free to query different
      service endpoints for different API versions. However, bots connectors
      and webhook calls will follow the specified API version.
    avatarUri: Optional. The URI of the agent's avatar. Avatars are used
      throughout the Dialogflow console and in the self-hosted [Web
      Demo](https://cloud.google.com/dialogflow/docs/integrations/web-demo)
      integration.
    classificationThreshold: Optional. To filter out false positive results
      and still get variety in matched natural language inputs for your agent,
      you can tune the machine learning classification threshold. If the
      returned score value is less than the threshold value, then a fallback
      intent will be triggered or, if there are no fallback intents defined,
      no intent will be triggered. The score values range from 0.0 (completely
      uncertain) to 1.0 (completely certain). If set to 0.0, the default of
      0.3 is used.
    defaultLanguageCode: Required. The default language of the agent as a
      language tag. See [Language
      Support](https://cloud.google.com/dialogflow/docs/reference/language)
      for a list of the currently supported language codes. This field cannot
      be set by the `Update` method.
    description: Optional. The description of this agent. The maximum length
      is 500 characters. If exceeded, the request is rejected.
    displayName: Required. The name of this agent.
    enableLogging: Optional. Determines whether this agent should log
      conversation queries.
    matchMode: Optional. Determines how intents are detected from user
      queries.
    parent: Required. The project of this agent. Format: `projects/`.
    supportedLanguageCodes: Optional. The list of all languages supported by
      this agent (except for the `default_language_code`).
    tier: Optional. The agent tier. If not specified, TIER_STANDARD is
      assumed.
    timeZone: Required. The time zone of this agent from the [time zone
      database](https://www.iana.org/time-zones), e.g., America/New_York,
      Europe/Paris.
  """

    class ApiVersionValueValuesEnum(_messages.Enum):
        """Optional. API version displayed in Dialogflow console. If not
    specified, V2 API is assumed. Clients are free to query different service
    endpoints for different API versions. However, bots connectors and webhook
    calls will follow the specified API version.

    Values:
      API_VERSION_UNSPECIFIED: Not specified.
      API_VERSION_V1: Legacy V1 API.
      API_VERSION_V2: V2 API.
      API_VERSION_V2_BETA_1: V2beta1 API.
    """
        API_VERSION_UNSPECIFIED = 0
        API_VERSION_V1 = 1
        API_VERSION_V2 = 2
        API_VERSION_V2_BETA_1 = 3

    class MatchModeValueValuesEnum(_messages.Enum):
        """Optional. Determines how intents are detected from user queries.

    Values:
      MATCH_MODE_UNSPECIFIED: Not specified.
      MATCH_MODE_HYBRID: Best for agents with a small number of examples in
        intents and/or wide use of templates syntax and composite entities.
      MATCH_MODE_ML_ONLY: Can be used for agents with a large number of
        examples in intents, especially the ones using @sys.any or very large
        custom entities.
    """
        MATCH_MODE_UNSPECIFIED = 0
        MATCH_MODE_HYBRID = 1
        MATCH_MODE_ML_ONLY = 2

    class TierValueValuesEnum(_messages.Enum):
        """Optional. The agent tier. If not specified, TIER_STANDARD is assumed.

    Values:
      TIER_UNSPECIFIED: Not specified. This value should never be used.
      TIER_STANDARD: Trial Edition, previously known as Standard Edition.
      TIER_ENTERPRISE: Essentials Edition, previously known as Enterprise
        Essential Edition.
      TIER_ENTERPRISE_PLUS: Essentials Edition (same as TIER_ENTERPRISE),
        previously known as Enterprise Plus Edition.
    """
        TIER_UNSPECIFIED = 0
        TIER_STANDARD = 1
        TIER_ENTERPRISE = 2
        TIER_ENTERPRISE_PLUS = 3
    apiVersion = _messages.EnumField('ApiVersionValueValuesEnum', 1)
    avatarUri = _messages.StringField(2)
    classificationThreshold = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    defaultLanguageCode = _messages.StringField(4)
    description = _messages.StringField(5)
    displayName = _messages.StringField(6)
    enableLogging = _messages.BooleanField(7)
    matchMode = _messages.EnumField('MatchModeValueValuesEnum', 8)
    parent = _messages.StringField(9)
    supportedLanguageCodes = _messages.StringField(10, repeated=True)
    tier = _messages.EnumField('TierValueValuesEnum', 11)
    timeZone = _messages.StringField(12)