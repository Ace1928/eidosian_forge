from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TreeMethodValueValuesEnum(_messages.Enum):
    """Tree construction algorithm for boosted tree models.

    Values:
      TREE_METHOD_UNSPECIFIED: Unspecified tree method.
      AUTO: Use heuristic to choose the fastest method.
      EXACT: Exact greedy algorithm.
      APPROX: Approximate greedy algorithm using quantile sketch and gradient
        histogram.
      HIST: Fast histogram optimized approximate greedy algorithm.
    """
    TREE_METHOD_UNSPECIFIED = 0
    AUTO = 1
    EXACT = 2
    APPROX = 3
    HIST = 4