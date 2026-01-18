from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsOperationsPatchRequest(_messages.Message):
    """A ContaineranalysisProjectsOperationsPatchRequest object.

  Fields:
    name: The name of the Operation. Should be of the form
      "projects/{provider_id}/operations/{operation_id}".
    updateOperationRequest: A UpdateOperationRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    updateOperationRequest = _messages.MessageField('UpdateOperationRequest', 2)