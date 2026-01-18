from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GenerateContentResponse(_messages.Message):
    """Response message for [PredictionService.GenerateContent].

  Fields:
    candidates: Output only. Generated candidates.
    promptFeedback: Output only. Content filter results for a prompt sent in
      the request. Note: Sent only in the first stream chunk. Only happens
      when no candidates were generated due to content violations.
    usageMetadata: Usage metadata about the response(s).
  """
    candidates = _messages.MessageField('GoogleCloudAiplatformV1beta1Candidate', 1, repeated=True)
    promptFeedback = _messages.MessageField('GoogleCloudAiplatformV1beta1GenerateContentResponsePromptFeedback', 2)
    usageMetadata = _messages.MessageField('GoogleCloudAiplatformV1beta1GenerateContentResponseUsageMetadata', 3)