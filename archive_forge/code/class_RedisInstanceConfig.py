from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisInstanceConfig(_messages.Message):
    """Message for Redis instance configs.

  Fields:
    memory_size_gb: The redis instance memory size, in GB.
  """
    memory_size_gb = _messages.IntegerField(1, variant=_messages.Variant.INT32)