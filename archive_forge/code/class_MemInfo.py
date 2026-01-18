from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemInfo(_messages.Message):
    """Information about the memory usage of a worker or a container within a
  worker.

  Fields:
    currentLimitBytes: Instantenous memory limit in bytes.
    currentOoms: Number of Out of Memory (OOM) events recorded since the
      previous measurement.
    currentRssBytes: Instantenous memory (RSS) size in bytes.
    timestamp: Timestamp of the measurement.
    totalGbMs: Total memory (RSS) usage since start up in GB * ms.
  """
    currentLimitBytes = _messages.IntegerField(1, variant=_messages.Variant.UINT64)
    currentOoms = _messages.IntegerField(2)
    currentRssBytes = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    timestamp = _messages.StringField(4)
    totalGbMs = _messages.IntegerField(5, variant=_messages.Variant.UINT64)