from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IngestMessageResponse(_messages.Message):
    """Acknowledges that a message has been ingested into the specified HL7v2
  store.

  Fields:
    hl7Ack: HL7v2 ACK message.
    message: Created message resource.
  """
    hl7Ack = _messages.BytesField(1)
    message = _messages.MessageField('Message', 2)