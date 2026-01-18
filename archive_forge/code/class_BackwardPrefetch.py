from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
class BackwardPrefetch(Enum):
    """
    This configures explicit backward prefetching, which improves throughput by
    enabling communication and computation overlap in the backward pass at the
    cost of slightly increased memory usage.

    - ``BACKWARD_PRE``: This enables the most overlap but increases memory
      usage the most. This prefetches the next set of parameters *before* the
      current set of parameters' gradient computation. This overlaps the *next
      all-gather* and the *current gradient computation*, and at the peak, it
      holds the current set of parameters, next set of parameters, and current
      set of gradients in memory.
    - ``BACKWARD_POST``: This enables less overlap but requires less memory
      usage. This prefetches the next set of parameters *after* the current
      set of parameters' gradient computation. This overlaps the *current
      reduce-scatter* and the *next gradient computation*, and it frees the
      current set of parameters before allocating memory for the next set of
      parameters, only holding the next set of parameters and current set of
      gradients in memory at the peak.
    - FSDP's ``backward_prefetch`` argument accepts ``None``, which disables
      the backward prefetching altogether. This has no overlap and does not
      increase memory usage. In general, we do not recommend this setting since
      it may degrade throughput significantly.

    For more technical context: For a single process group using NCCL backend,
    any collectives, even if issued from different streams, contend for the
    same per-device NCCL stream, which implies that the relative order in which
    the collectives are issued matters for overlapping. The two backward
    prefetching values correspond to different issue orders.
    """
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()