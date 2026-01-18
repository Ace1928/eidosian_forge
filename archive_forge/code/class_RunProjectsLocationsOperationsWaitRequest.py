from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsOperationsWaitRequest(_messages.Message):
    """A RunProjectsLocationsOperationsWaitRequest object.

  Fields:
    googleLongrunningWaitOperationRequest: A
      GoogleLongrunningWaitOperationRequest resource to be passed as the
      request body.
    name: The name of the operation resource to wait on.
  """
    googleLongrunningWaitOperationRequest = _messages.MessageField('GoogleLongrunningWaitOperationRequest', 1)
    name = _messages.StringField(2, required=True)