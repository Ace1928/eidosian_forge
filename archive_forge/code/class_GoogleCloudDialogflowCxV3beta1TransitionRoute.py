from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1TransitionRoute(_messages.Message):
    """A transition route specifies a intent that can be matched and/or a data
  condition that can be evaluated during a session. When a specified
  transition is matched, the following actions are taken in order: * If there
  is a `trigger_fulfillment` associated with the transition, it will be
  called. * If there is a `target_page` associated with the transition, the
  session will transition into the specified page. * If there is a
  `target_flow` associated with the transition, the session will transition
  into the specified flow.

  Fields:
    condition: The condition to evaluate against form parameters or session
      parameters. See the [conditions reference](https://cloud.google.com/dial
      ogflow/cx/docs/reference/condition). At least one of `intent` or
      `condition` must be specified. When both `intent` and `condition` are
      specified, the transition can only happen when both are fulfilled.
    description: Optional. The description of the transition route. The
      maximum length is 500 characters.
    intent: The unique identifier of an Intent. Format:
      `projects//locations//agents//intents/`. Indicates that the transition
      can only happen when the given intent is matched. At least one of
      `intent` or `condition` must be specified. When both `intent` and
      `condition` are specified, the transition can only happen when both are
      fulfilled.
    name: Output only. The unique identifier of this transition route.
    targetFlow: The target flow to transition to. Format:
      `projects//locations//agents//flows/`.
    targetPage: The target page to transition to. Format:
      `projects//locations//agents//flows//pages/`.
    triggerFulfillment: The fulfillment to call when the condition is
      satisfied. At least one of `trigger_fulfillment` and `target` must be
      specified. When both are defined, `trigger_fulfillment` is executed
      first.
  """
    condition = _messages.StringField(1)
    description = _messages.StringField(2)
    intent = _messages.StringField(3)
    name = _messages.StringField(4)
    targetFlow = _messages.StringField(5)
    targetPage = _messages.StringField(6)
    triggerFulfillment = _messages.MessageField('GoogleCloudDialogflowCxV3beta1Fulfillment', 7)