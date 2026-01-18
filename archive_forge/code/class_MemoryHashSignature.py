from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemoryHashSignature(_messages.Message):
    """A signature corresponding to memory page hashes.

  Fields:
    binaryFamily: The binary family.
    detections: The list of memory hash detections contributing to the binary
      family match.
  """
    binaryFamily = _messages.StringField(1)
    detections = _messages.MessageField('Detection', 2, repeated=True)