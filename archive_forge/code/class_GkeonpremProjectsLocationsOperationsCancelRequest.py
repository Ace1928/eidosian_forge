from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkeonpremProjectsLocationsOperationsCancelRequest(_messages.Message):
    """A GkeonpremProjectsLocationsOperationsCancelRequest object.

  Fields:
    cancelOperationRequest: A CancelOperationRequest resource to be passed as
      the request body.
    name: The name of the operation resource to be cancelled.
  """
    cancelOperationRequest = _messages.MessageField('CancelOperationRequest', 1)
    name = _messages.StringField(2, required=True)