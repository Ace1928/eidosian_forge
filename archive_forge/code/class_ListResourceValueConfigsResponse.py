from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListResourceValueConfigsResponse(_messages.Message):
    """Response message to list resource value configs

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is empty, there are no subsequent pages.
    resourceValueConfigs: The resource value configs from the specified
      parent.
  """
    nextPageToken = _messages.StringField(1)
    resourceValueConfigs = _messages.MessageField('GoogleCloudSecuritycenterV2ResourceValueConfig', 2, repeated=True)