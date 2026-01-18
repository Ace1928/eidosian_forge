from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListEndpointsResponse(_messages.Message):
    """Response message for EndpointService.ListEndpoints.

  Fields:
    endpoints: List of Endpoints in the requested page.
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListEndpointsRequest.page_token to obtain that page.
  """
    endpoints = _messages.MessageField('GoogleCloudAiplatformV1Endpoint', 1, repeated=True)
    nextPageToken = _messages.StringField(2)