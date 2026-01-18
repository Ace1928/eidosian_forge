from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BfdStatusPacketCounts(_messages.Message):
    """A BfdStatusPacketCounts object.

  Fields:
    numRx: Number of packets received since the beginning of the current BFD
      session.
    numRxRejected: Number of packets received that were rejected because of
      errors since the beginning of the current BFD session.
    numRxSuccessful: Number of packets received that were successfully
      processed since the beginning of the current BFD session.
    numTx: Number of packets transmitted since the beginning of the current
      BFD session.
  """
    numRx = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    numRxRejected = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    numRxSuccessful = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    numTx = _messages.IntegerField(4, variant=_messages.Variant.UINT32)