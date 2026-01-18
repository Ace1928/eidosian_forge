from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBitbucketServerConfigsResponse(_messages.Message):
    """RPC response object returned by ListBitbucketServerConfigs RPC method.

  Fields:
    bitbucketServerConfigs: A list of BitbucketServerConfigs
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    bitbucketServerConfigs = _messages.MessageField('BitbucketServerConfig', 1, repeated=True)
    nextPageToken = _messages.StringField(2)