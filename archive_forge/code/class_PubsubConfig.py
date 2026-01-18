from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubConfig(_messages.Message):
    """PubsubConfig describes the configuration of a trigger that creates a
  build whenever a Pub/Sub message is published.

  Enums:
    StateValueValuesEnum: Potential issues with the underlying Pub/Sub
      subscription configuration. Only populated on get requests.

  Fields:
    serviceAccountEmail: Service account that will make the push request.
    state: Potential issues with the underlying Pub/Sub subscription
      configuration. Only populated on get requests.
    subscription: Output only. Name of the subscription. Format is
      `projects/{project}/subscriptions/{subscription}`.
    topic: The name of the topic from which this subscription is receiving
      messages. Format is `projects/{project}/topics/{topic}`.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Potential issues with the underlying Pub/Sub subscription
    configuration. Only populated on get requests.

    Values:
      STATE_UNSPECIFIED: The subscription configuration has not been checked.
      OK: The Pub/Sub subscription is properly configured.
      SUBSCRIPTION_DELETED: The subscription has been deleted.
      TOPIC_DELETED: The topic has been deleted.
      SUBSCRIPTION_MISCONFIGURED: Some of the subscription's field are
        misconfigured.
    """
        STATE_UNSPECIFIED = 0
        OK = 1
        SUBSCRIPTION_DELETED = 2
        TOPIC_DELETED = 3
        SUBSCRIPTION_MISCONFIGURED = 4
    serviceAccountEmail = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    subscription = _messages.StringField(3)
    topic = _messages.StringField(4)