from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2LoggingConfig(_messages.Message):
    """Defines logging behavior for conversation lifecycle events.

  Fields:
    enableStackdriverLogging: Whether to log conversation events like
      CONVERSATION_STARTED to Stackdriver in the conversation project as JSON
      format ConversationEvent protos.
  """
    enableStackdriverLogging = _messages.BooleanField(1)