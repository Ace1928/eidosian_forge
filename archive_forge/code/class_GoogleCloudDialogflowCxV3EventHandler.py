from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3EventHandler(_messages.Message):
    """An event handler specifies an event that can be handled during a
  session. When the specified event happens, the following actions are taken
  in order: * If there is a `trigger_fulfillment` associated with the event,
  it will be called. * If there is a `target_page` associated with the event,
  the session will transition into the specified page. * If there is a
  `target_flow` associated with the event, the session will transition into
  the specified flow.

  Fields:
    event: Required. The name of the event to handle.
    name: Output only. The unique identifier of this event handler.
    targetFlow: The target flow to transition to. Format:
      `projects//locations//agents//flows/`.
    targetPage: The target page to transition to. Format:
      `projects//locations//agents//flows//pages/`.
    triggerFulfillment: The fulfillment to call when the event occurs.
      Handling webhook errors with a fulfillment enabled with webhook could
      cause infinite loop. It is invalid to specify such fulfillment for a
      handler handling webhooks.
  """
    event = _messages.StringField(1)
    name = _messages.StringField(2)
    targetFlow = _messages.StringField(3)
    targetPage = _messages.StringField(4)
    triggerFulfillment = _messages.MessageField('GoogleCloudDialogflowCxV3Fulfillment', 5)