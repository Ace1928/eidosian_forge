from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1betaListWorkloadIdentityPoolProvidersResponse(_messages.Message):
    """Response message for ListWorkloadIdentityPoolProviders.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    workloadIdentityPoolProviders: A list of providers.
  """
    nextPageToken = _messages.StringField(1)
    workloadIdentityPoolProviders = _messages.MessageField('GoogleIamV1betaWorkloadIdentityPoolProvider', 2, repeated=True)