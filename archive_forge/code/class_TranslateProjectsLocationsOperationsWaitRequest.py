from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsOperationsWaitRequest(_messages.Message):
    """A TranslateProjectsLocationsOperationsWaitRequest object.

  Fields:
    name: The name of the operation resource to wait on.
    waitOperationRequest: A WaitOperationRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    waitOperationRequest = _messages.MessageField('WaitOperationRequest', 2)