from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedisConfig(_messages.Message):
    """Message for Redis configs.

  Fields:
    instance: Configs for the Redis instance.
  """
    instance = _messages.MessageField('RedisInstanceConfig', 1)