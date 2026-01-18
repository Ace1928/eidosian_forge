from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProcessWorkflowTriggerWebhookRequest(_messages.Message):
    """Message for processing webhooks posted to WorkflowTrigger.

  Fields:
    body: Required. The webhook body in JSON.
    secretToken: Required. The secret token used for authorization based on
      the matching result between this and the secret stored in
      WorkflowTrigger.
    triggerId: Required. The WorkflowTrigger id.
  """
    body = _messages.MessageField('HttpBody', 1)
    secretToken = _messages.StringField(2)
    triggerId = _messages.StringField(3)