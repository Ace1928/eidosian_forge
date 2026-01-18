from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaListProxyConfigsResponse(_messages.Message):
    """Message for response to listing ProxyConfigs.

  Fields:
    nextPageToken: A token to retrieve the next page of results, or empty if
      there are no more results in the list.
    proxyConfigs: The list of ProxyConfig objects.
  """
    nextPageToken = _messages.StringField(1)
    proxyConfigs = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaProxyConfig', 2, repeated=True)