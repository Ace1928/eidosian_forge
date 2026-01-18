from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2IntentMessageSimpleResponses(_messages.Message):
    """The collection of simple response candidates. This message in
  `QueryResult.fulfillment_messages` and
  `WebhookResponse.fulfillment_messages` should contain only one
  `SimpleResponse`.

  Fields:
    simpleResponses: Required. The list of simple responses.
  """
    simpleResponses = _messages.MessageField('GoogleCloudDialogflowV2IntentMessageSimpleResponse', 1, repeated=True)