from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEnvironmentsIntentsListRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEnvironmentsIntentsListRequest object.

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
    pageSize: Optional. The maximum number of items to return in a single
      page. By default 100 and at most 1000.
    pageToken: Optional. The next_page_token value returned from a previous
      list request.
    parent: Required. The agent to list all intents from. Format:
      `projects//agent` or `projects//locations//agent`. Alternatively, you
      can specify the environment to list intents for. Format:
      `projects//agent/environments/` or
      `projects//locations//agent/environments/`. Note: training phrases of
      the intents will not be returned for non-draft environment.
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
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)