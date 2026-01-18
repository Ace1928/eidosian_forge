from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsBeginTransactionRequest(_messages.Message):
    """A DatastoreProjectsBeginTransactionRequest object.

  Fields:
    beginTransactionRequest: A BeginTransactionRequest resource to be passed
      as the request body.
    projectId: Required. The ID of the project against which to make the
      request.
  """
    beginTransactionRequest = _messages.MessageField('BeginTransactionRequest', 1)
    projectId = _messages.StringField(2, required=True)