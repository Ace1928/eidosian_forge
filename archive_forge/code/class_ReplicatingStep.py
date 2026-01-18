from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicatingStep(_messages.Message):
    """ReplicatingStep contains specific step details.

  Fields:
    lastThirtyMinutesAverageBytesPerSecond: The source disks replication rate
      for the last 30 minutes in bytes per second.
    lastTwoMinutesAverageBytesPerSecond: The source disks replication rate for
      the last 2 minutes in bytes per second.
    replicatedBytes: Replicated bytes in the step.
    totalBytes: Total bytes to be handled in the step.
  """
    lastThirtyMinutesAverageBytesPerSecond = _messages.IntegerField(1)
    lastTwoMinutesAverageBytesPerSecond = _messages.IntegerField(2)
    replicatedBytes = _messages.IntegerField(3)
    totalBytes = _messages.IntegerField(4)