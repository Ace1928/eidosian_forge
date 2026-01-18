from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Hl7V2StoreMetric(_messages.Message):
    """Count of messages and total storage size by type for a given HL7 store.

  Fields:
    count: The total count of HL7v2 messages in the store for the given
      message type.
    messageType: The Hl7v2 message type this metric applies to, such as `ADT`
      or `ORU`.
    structuredStorageSizeBytes: The total amount of structured storage used by
      HL7v2 messages of this message type in the store.
  """
    count = _messages.IntegerField(1)
    messageType = _messages.StringField(2)
    structuredStorageSizeBytes = _messages.IntegerField(3)