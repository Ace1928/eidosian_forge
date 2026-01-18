from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationModel(_messages.Message):
    """Represents a conversation model.

  Enums:
    StateValueValuesEnum: Output only. State of the model. A model can only
      serve prediction requests after it gets deployed.

  Fields:
    articleSuggestionModelMetadata: Metadata for article suggestion models.
    createTime: Output only. Creation time of this model.
    datasets: Required. Datasets used to create model.
    displayName: Required. The display name of the model. At most 64 bytes
      long.
    languageCode: Language code for the conversation model. If not specified,
      the language is en-US. Language at ConversationModel should be set for
      all non en-us languages. This should be a [BCP-47](https://www.rfc-
      editor.org/rfc/bcp/bcp47.txt) language tag. Example: "en-US".
    name: ConversationModel resource name. Format:
      `projects//conversationModels/`
    smartReplyModelMetadata: Metadata for smart reply models.
    state: Output only. State of the model. A model can only serve prediction
      requests after it gets deployed.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the model. A model can only serve prediction
    requests after it gets deployed.

    Values:
      STATE_UNSPECIFIED: Should not be used, an un-set enum has this value by
        default.
      CREATING: Model being created.
      UNDEPLOYED: Model is not deployed but ready to deploy.
      DEPLOYING: Model is deploying.
      DEPLOYED: Model is deployed and ready to use.
      UNDEPLOYING: Model is undeploying.
      DELETING: Model is deleting.
      FAILED: Model is in error state. Not ready to deploy and use.
      PENDING: Model is being created but the training has not started, The
        model may remain in this state until there is enough capacity to start
        training.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        UNDEPLOYED = 2
        DEPLOYING = 3
        DEPLOYED = 4
        UNDEPLOYING = 5
        DELETING = 6
        FAILED = 7
        PENDING = 8
    articleSuggestionModelMetadata = _messages.MessageField('GoogleCloudDialogflowV2ArticleSuggestionModelMetadata', 1)
    createTime = _messages.StringField(2)
    datasets = _messages.MessageField('GoogleCloudDialogflowV2InputDataset', 3, repeated=True)
    displayName = _messages.StringField(4)
    languageCode = _messages.StringField(5)
    name = _messages.StringField(6)
    smartReplyModelMetadata = _messages.MessageField('GoogleCloudDialogflowV2SmartReplyModelMetadata', 7)
    state = _messages.EnumField('StateValueValuesEnum', 8)