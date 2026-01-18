from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGithubEnterpriseConfigsResponse(_messages.Message):
    """RPC response object returned by ListGithubEnterpriseConfigs RPC method.

  Fields:
    configs: A list of GitHubEnterpriseConfigs
  """
    configs = _messages.MessageField('GitHubEnterpriseConfig', 1, repeated=True)