from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateBitbucketServerConnectedRepositoryRequest(_messages.Message):
    """Request to connect a repository from a connected Bitbucket Server host.

  Fields:
    bitbucketServerConnectedRepository: Required. The Bitbucket Server
      repository to connect.
    parent: Required. The name of the `BitbucketServerConfig` that added
      connected repository. Format: `projects/{project}/locations/{location}/b
      itbucketServerConfigs/{config}`
  """
    bitbucketServerConnectedRepository = _messages.MessageField('BitbucketServerConnectedRepository', 1)
    parent = _messages.StringField(2)