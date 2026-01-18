from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferStats(_messages.Message):
    """TransferStats reports all statistics related to replication transfer.

  Fields:
    lagDuration: Lag duration indicates the duration by which Destination
      region volume content lags behind the primary region volume content.
    lastTransferBytes: Last transfer size in bytes.
    lastTransferDuration: Time taken during last transfer.
    lastTransferEndTime: Time when last transfer completed.
    lastTransferError: A message describing the cause of the last transfer
      failure.
    totalTransferDuration: Total time taken during transfer.
    transferBytes: bytes trasferred so far in current transfer.
    updateTime: Time when progress was updated last.
  """
    lagDuration = _messages.StringField(1)
    lastTransferBytes = _messages.IntegerField(2)
    lastTransferDuration = _messages.StringField(3)
    lastTransferEndTime = _messages.StringField(4)
    lastTransferError = _messages.StringField(5)
    totalTransferDuration = _messages.StringField(6)
    transferBytes = _messages.IntegerField(7)
    updateTime = _messages.StringField(8)