from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListModelEvaluationsResponse(_messages.Message):
    """Response message for ModelService.ListModelEvaluations.

  Fields:
    modelEvaluations: List of ModelEvaluations in the requested page.
    nextPageToken: A token to retrieve next page of results. Pass to
      ListModelEvaluationsRequest.page_token to obtain that page.
  """
    modelEvaluations = _messages.MessageField('GoogleCloudAiplatformV1beta1ModelEvaluation', 1, repeated=True)
    nextPageToken = _messages.StringField(2)