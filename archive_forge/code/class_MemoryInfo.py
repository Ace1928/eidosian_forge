from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemoryInfo(_messages.Message):
    """A MemoryInfo object.

  Fields:
    memoryCapInKibibyte: Maximum memory that can be allocated to the process
      in KiB
    memoryTotalInKibibyte: Total memory available on the device in KiB
  """
    memoryCapInKibibyte = _messages.IntegerField(1)
    memoryTotalInKibibyte = _messages.IntegerField(2)