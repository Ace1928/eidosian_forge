from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubProjectsSubscriptionsModifyAckDeadlineRequest(_messages.Message):
    """A PubsubProjectsSubscriptionsModifyAckDeadlineRequest object.

  Fields:
    modifyAckDeadlineRequest: A ModifyAckDeadlineRequest resource to be passed
      as the request body.
    subscription: The name of the subscription. Format is
      `projects/{project}/subscriptions/{sub}`.
  """
    modifyAckDeadlineRequest = _messages.MessageField('ModifyAckDeadlineRequest', 1)
    subscription = _messages.StringField(2, required=True)