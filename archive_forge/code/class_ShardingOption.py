from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ShardingOption(_messages.Message):
    """Options for enabling sharding.

  Fields:
    manualSharding: Shards test cases into the specified groups of packages,
      classes, and/or methods.
    smartSharding: Shards test based on previous test case timing records.
    uniformSharding: Uniformly shards test cases given a total number of
      shards.
  """
    manualSharding = _messages.MessageField('ManualSharding', 1)
    smartSharding = _messages.MessageField('SmartSharding', 2)
    uniformSharding = _messages.MessageField('UniformSharding', 3)