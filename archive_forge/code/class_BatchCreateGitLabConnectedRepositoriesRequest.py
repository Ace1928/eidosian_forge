from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateGitLabConnectedRepositoriesRequest(_messages.Message):
    """RPC request object accepted by BatchCreateGitLabConnectedRepositories
  RPC method.

  Fields:
    requests: Required. Requests to connect GitLab repositories.
  """
    requests = _messages.MessageField('CreateGitLabConnectedRepositoryRequest', 1, repeated=True)