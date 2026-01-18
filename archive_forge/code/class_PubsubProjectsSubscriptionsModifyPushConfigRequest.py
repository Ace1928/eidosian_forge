from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsModifyPushConfigRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsModifyPushConfigRequest object.

  Fields:
    modifyPushConfigRequest: A ModifyPushConfigRequest resource to be passed
      as the request body.
    subscription: The name of the subscription. Format is
      `projects/{project}/subscriptions/{sub}`.
  """
    modifyPushConfigRequest = _messages.MessageField('ModifyPushConfigRequest', 1)
    subscription = _messages.StringField(2, required=True)