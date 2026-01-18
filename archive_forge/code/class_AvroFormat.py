from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AvroFormat(_messages.Message):
    """Configuration for reading Cloud Storage data in Avro binary format. The
  bytes of each object will be set to the `data` field of a Pub/Sub message.
  """