import collections
import copy
import enum
import inspect
import io
import logging
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer
class _ZeROJoinHook(JoinHook):

    def __init__(self, zero):
        assert isinstance(zero, ZeroRedundancyOptimizer), 'ZeRO join hook requires passing in a ZeroRedundancyOptimizer instance as the state'
        self.zero = zero
        super().__init__()

    def main_hook(self):
        """
        Perform an optimizer step.

        This step updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """
        self.zero.step()