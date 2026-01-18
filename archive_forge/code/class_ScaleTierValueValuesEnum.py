from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScaleTierValueValuesEnum(_messages.Enum):
    """Required. Scale tier of the hardware used for notebook execution.
    DEPRECATED Will be discontinued. As right now only CUSTOM is supported.

    Values:
      SCALE_TIER_UNSPECIFIED: Unspecified Scale Tier.
      BASIC: A single worker instance. This tier is suitable for learning how
        to use Cloud ML, and for experimenting with new models using small
        datasets.
      STANDARD_1: Many workers and a few parameter servers.
      PREMIUM_1: A large number of workers with many parameter servers.
      BASIC_GPU: A single worker instance with a K80 GPU.
      BASIC_TPU: A single worker instance with a Cloud TPU.
      CUSTOM: The CUSTOM tier is not a set tier, but rather enables you to use
        your own cluster specification. When you use this tier, set values to
        configure your processing cluster according to these guidelines: * You
        _must_ set `ExecutionTemplate.masterType` to specify the type of
        machine to use for your master node. This is the only required
        setting.
    """
    SCALE_TIER_UNSPECIFIED = 0
    BASIC = 1
    STANDARD_1 = 2
    PREMIUM_1 = 3
    BASIC_GPU = 4
    BASIC_TPU = 5
    CUSTOM = 6