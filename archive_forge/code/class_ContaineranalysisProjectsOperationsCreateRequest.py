from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsOperationsCreateRequest(_messages.Message):
    """A ContaineranalysisProjectsOperationsCreateRequest object.

  Fields:
    createOperationRequest: A CreateOperationRequest resource to be passed as
      the request body.
    parent: The project Id that this operation should be created under.
  """
    createOperationRequest = _messages.MessageField('CreateOperationRequest', 1)
    parent = _messages.StringField(2, required=True)