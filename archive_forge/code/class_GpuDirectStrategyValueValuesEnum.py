from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GpuDirectStrategyValueValuesEnum(_messages.Enum):
    """The type of GPU direct strategy to enable on the node pool.

    Values:
      GPU_DIRECT_STRATEGY_UNSPECIFIED: Default value. No GPU Direct strategy
        is enabled on the node.
      TCPX: GPUDirect-TCPX on A3
    """
    GPU_DIRECT_STRATEGY_UNSPECIFIED = 0
    TCPX = 1