from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ModifyPushConfigRequest(_messages.Message):
    """Request for the ModifyPushConfig method.

  Fields:
    pushConfig: The push configuration for future deliveries.  An empty
      `pushConfig` indicates that the Pub/Sub system should stop pushing
      messages from the given subscription and allow messages to be pulled and
      acknowledged - effectively pausing the subscription if `Pull` or
      `StreamingPull` is not called.
  """
    pushConfig = _messages.MessageField('PushConfig', 1)