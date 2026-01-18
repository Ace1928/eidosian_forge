from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentIntentsCreateRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentIntentsCreateRequest object.

  Enums:
    IntentViewValueValuesEnum: Optional. The resource view to apply to the
      returned intent.

  Fields:
    googleCloudDialogflowV2Intent: A GoogleCloudDialogflowV2Intent resource to
      be passed as the request body.
    intentView: Optional. The resource view to apply to the returned intent.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    parent: Required. The agent to create a intent for. Format:
      `projects//agent`.
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
    googleCloudDialogflowV2Intent = _messages.MessageField('GoogleCloudDialogflowV2Intent', 1)
    intentView = _messages.EnumField('IntentViewValueValuesEnum', 2)
    languageCode = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)