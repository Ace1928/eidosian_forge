from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ConversationPhoneNumber(_messages.Message):
    """Represents a phone number for telephony integration. It allows for
  connecting a particular conversation over telephony.

  Fields:
    phoneNumber: Output only. The phone number to connect to this
      conversation.
  """
    phoneNumber = _messages.StringField(1)