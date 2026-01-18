from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BitbucketServerConnectedRepository(_messages.Message):
    """/ BitbucketServerConnectedRepository represents a connected Bitbucket
  Server / repository.

  Fields:
    parent: The name of the `BitbucketServerConfig` that added connected
      repository. Format: `projects/{project}/locations/{location}/bitbucketSe
      rverConfigs/{config}`
    repo: The Bitbucket Server repositories to connect.
    status: Output only. The status of the repo connection request.
  """
    parent = _messages.StringField(1)
    repo = _messages.MessageField('BitbucketServerRepositoryId', 2)
    status = _messages.MessageField('Status', 3)