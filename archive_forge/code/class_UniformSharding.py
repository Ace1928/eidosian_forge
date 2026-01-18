from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class UniformSharding(_messages.Message):
    """Uniformly shards test cases given a total number of shards. For
  instrumentation tests, it will be translated to "-e numShard" and "-e
  shardIndex" AndroidJUnitRunner arguments. With uniform sharding enabled,
  specifying either of these sharding arguments via `environment_variables` is
  invalid. Based on the sharding mechanism AndroidJUnitRunner uses, there is
  no guarantee that test cases will be distributed uniformly across all
  shards.

  Fields:
    numShards: Required. The total number of shards to create. This must
      always be a positive number that is no greater than the total number of
      test cases. When you select one or more physical devices, the number of
      shards must be <= 50. When you select one or more ARM virtual devices,
      it must be <= 200. When you select only x86 virtual devices, it must be
      <= 500.
  """
    numShards = _messages.IntegerField(1, variant=_messages.Variant.INT32)