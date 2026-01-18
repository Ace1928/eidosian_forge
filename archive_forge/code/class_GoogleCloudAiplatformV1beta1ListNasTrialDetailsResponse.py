from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListNasTrialDetailsResponse(_messages.Message):
    """Response message for JobService.ListNasTrialDetails

  Fields:
    nasTrialDetails: List of top NasTrials in the requested page.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListNasTrialDetailsRequest.page_token to obtain that page.
  """
    nasTrialDetails = _messages.MessageField('GoogleCloudAiplatformV1beta1NasTrialDetail', 1, repeated=True)
    nextPageToken = _messages.StringField(2)