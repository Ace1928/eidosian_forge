import copy
import os
def _ShardName(name, number):
    """Add a shard number to the end of a target.

  Arguments:
    name: name of the target (foo#target)
    number: shard number
  Returns:
    Target name with shard added (foo_1#target)
  """
    return _SuffixName(name, str(number))