import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class vote_sync_intrinsic(Stub):
    """
    vote_sync_intrinsic(mask, mode, predictate)

    Nvvm intrinsic for performing a reduce and broadcast across a warp
    docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-vote
    """
    _description_ = '<vote_sync()>'