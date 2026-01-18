from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TriggerPubsubExecutionRequest(_messages.Message):
    """Request for the TriggerPubsubExecution method.

  Fields:
    GCPCloudEventsMode: Required. LINT: LEGACY_NAMES The query parameter value
      for __GCP_CloudEventsMode, set by the Eventarc service when configuring
      triggers.
    deliveryAttempt: The number of attempts that have been made to deliver
      this message. This is set by Pub/Sub for subscriptions that have the
      "dead letter" feature enabled, and hence provided here for
      compatibility, but is ignored by Workflows.
    message: Required. The message of the Pub/Sub push notification.
    subscription: Required. The subscription of the Pub/Sub push notification.
      Format: projects/{project}/subscriptions/{sub}
  """
    GCPCloudEventsMode = _messages.StringField(1)
    deliveryAttempt = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    message = _messages.MessageField('PubsubMessage', 3)
    subscription = _messages.StringField(4)