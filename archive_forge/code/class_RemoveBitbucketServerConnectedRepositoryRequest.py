from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoveBitbucketServerConnectedRepositoryRequest(_messages.Message):
    """RPC request object accepted by RemoveBitbucketServerConnectedRepository
  RPC method.

  Fields:
    connectedRepository: The connected repository to remove.
  """
    connectedRepository = _messages.MessageField('BitbucketServerRepositoryId', 1)