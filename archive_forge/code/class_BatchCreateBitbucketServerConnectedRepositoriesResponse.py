from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateBitbucketServerConnectedRepositoriesResponse(_messages.Message):
    """Response of BatchCreateBitbucketServerConnectedRepositories RPC method
  including all successfully connected Bitbucket Server repositories.

  Fields:
    bitbucketServerConnectedRepositories: The connected Bitbucket Server
      repositories.
  """
    bitbucketServerConnectedRepositories = _messages.MessageField('BitbucketServerConnectedRepository', 1, repeated=True)