from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListReasoningEnginesResponse(_messages.Message):
    """Response message for ReasoningEngineService.ListReasoningEngines

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListReasoningEnginesRequest.page_token to obtain that page.
    reasoningEngines: List of ReasoningEngines in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    reasoningEngines = _messages.MessageField('GoogleCloudAiplatformV1beta1ReasoningEngine', 2, repeated=True)