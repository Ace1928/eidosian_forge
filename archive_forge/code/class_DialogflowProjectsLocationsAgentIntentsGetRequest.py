from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentIntentsGetRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentIntentsGetRequest object.

  Enums:
    IntentViewValueValuesEnum: Optional. The resource view to apply to the
      returned intent.

  Fields:
    intentView: Optional. The resource view to apply to the returned intent.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    name: Required. The name of the intent. Format:
      `projects//agent/intents/`.
  """

    class IntentViewValueValuesEnum(_messages.Enum):
        """Optional. The resource view to apply to the returned intent.

    Values:
      INTENT_VIEW_UNSPECIFIED: Training phrases field is not populated in the
        response.
      INTENT_VIEW_FULL: All fields are populated.
    """
        INTENT_VIEW_UNSPECIFIED = 0
        INTENT_VIEW_FULL = 1
    intentView = _messages.EnumField('IntentViewValueValuesEnum', 1)
    languageCode = _messages.StringField(2)
    name = _messages.StringField(3, required=True)