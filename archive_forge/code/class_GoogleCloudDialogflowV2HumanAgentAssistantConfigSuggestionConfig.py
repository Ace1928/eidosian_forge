from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionConfig(_messages.Message):
    """Detail human agent assistant config.

  Fields:
    featureConfigs: Configuration of different suggestion features. One
      feature can have only one config.
    groupSuggestionResponses: If `group_suggestion_responses` is false, and
      there are multiple `feature_configs` in `event based suggestion` or
      StreamingAnalyzeContent, we will try to deliver suggestions to customers
      as soon as we get new suggestion. Different type of suggestions based on
      the same context will be in separate Pub/Sub event or
      `StreamingAnalyzeContentResponse`. If `group_suggestion_responses` set
      to true. All the suggestions to the same participant based on the same
      context will be grouped into a single Pub/Sub event or
      StreamingAnalyzeContentResponse.
  """
    featureConfigs = _messages.MessageField('GoogleCloudDialogflowV2HumanAgentAssistantConfigSuggestionFeatureConfig', 1, repeated=True)
    groupSuggestionResponses = _messages.BooleanField(2)