from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchUpdateIntentsRequest(_messages.Message):
    """A GoogleCloudDialogflowV2BatchUpdateIntentsRequest object.

  Enums:
    IntentViewValueValuesEnum: Optional. The resource view to apply to the
      returned intent.

  Fields:
    intentBatchInline: The collection of intents to update or create.
    intentBatchUri: The URI to a Google Cloud Storage file containing intents
      to update or create. The file format can either be a serialized proto
      (of IntentBatch type) or JSON object. Note: The URI must start with
      "gs://".
    intentView: Optional. The resource view to apply to the returned intent.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    updateMask: Optional. The mask to control which fields get updated.
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
    intentBatchInline = _messages.MessageField('GoogleCloudDialogflowV2IntentBatch', 1)
    intentBatchUri = _messages.StringField(2)
    intentView = _messages.EnumField('IntentViewValueValuesEnum', 3)
    languageCode = _messages.StringField(4)
    updateMask = _messages.StringField(5)