from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsCommitRequest(_messages.Message):
    """A DatastoreProjectsCommitRequest object.

  Fields:
    commitRequest: A CommitRequest resource to be passed as the request body.
    projectId: Required. The ID of the project against which to make the
      request.
  """
    commitRequest = _messages.MessageField('CommitRequest', 1)
    projectId = _messages.StringField(2, required=True)