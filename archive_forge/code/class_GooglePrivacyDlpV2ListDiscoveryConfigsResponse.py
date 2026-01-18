from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListDiscoveryConfigsResponse(_messages.Message):
    """Response message for ListDiscoveryConfigs.

  Fields:
    discoveryConfigs: List of configs, up to page_size in
      ListDiscoveryConfigsRequest.
    nextPageToken: If the next page is available then this value is the next
      page token to be used in the following ListDiscoveryConfigs request.
  """
    discoveryConfigs = _messages.MessageField('GooglePrivacyDlpV2DiscoveryConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)