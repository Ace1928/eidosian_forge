from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListGitHubInstallationsForProjectResponse(_messages.Message):
    """RPC response object returned by the ListGitHubInstallations RPC method.

  Fields:
    installations: Installations belonging to the specified project_id.
  """
    installations = _messages.MessageField('Installation', 1, repeated=True)