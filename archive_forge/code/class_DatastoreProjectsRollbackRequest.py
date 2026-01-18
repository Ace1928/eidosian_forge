from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsRollbackRequest(_messages.Message):
    """A DatastoreProjectsRollbackRequest object.

  Fields:
    projectId: Required. The ID of the project against which to make the
      request.
    rollbackRequest: A RollbackRequest resource to be passed as the request
      body.
  """
    projectId = _messages.StringField(1, required=True)
    rollbackRequest = _messages.MessageField('RollbackRequest', 2)