from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateMessageRequest(_messages.Message):
    """Creates a new message.

  Fields:
    message: HL7v2 message.
  """
    message = _messages.MessageField('Message', 1)