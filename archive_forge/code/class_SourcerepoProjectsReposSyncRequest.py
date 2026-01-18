from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposSyncRequest(_messages.Message):
    """A SourcerepoProjectsReposSyncRequest object.

  Fields:
    name: The name of the repo to synchronize. Values are of the form
      `projects//repos/`.
    syncRepoRequest: A SyncRepoRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    syncRepoRequest = _messages.MessageField('SyncRepoRequest', 2)