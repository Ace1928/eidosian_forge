from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesExecutionsCancelRequest(_messages.Message):
    """A RunNamespacesExecutionsCancelRequest object.

  Fields:
    cancelExecutionRequest: A CancelExecutionRequest resource to be passed as
      the request body.
    name: Required. The name of the execution to cancel. Replace {namespace}
      with the project ID or number. It takes the form namespaces/{namespace}.
      For example: namespaces/PROJECT_ID
  """
    cancelExecutionRequest = _messages.MessageField('CancelExecutionRequest', 1)
    name = _messages.StringField(2, required=True)