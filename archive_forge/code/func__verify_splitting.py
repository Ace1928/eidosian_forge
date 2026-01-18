from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast
import torch
from torch import Tensor, nn
from torch.distributed.rpc import RRef
import torch.autograd
import torch.cuda
from . import microbatch
from .batchnorm import DeferredBatchNorm
from .pipeline import Pipeline
from .skip.layout import inspect_skip_layout
from .skip.skippable import verify_skippables
from .stream import AbstractStream, new_stream
def _verify_splitting(module: nn.Sequential, partitions: List[nn.Sequential], devices: List[torch.device]) -> None:
    num_parameters = len(list(module.parameters()))
    num_child_parameters = sum((len(list(child.parameters())) for child in module.children()))
    if num_parameters == num_child_parameters:
        return
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            parti = partitions[i]
            partj = partitions[j]
            if devices[i] == devices[j]:
                continue
            for p in parti.parameters():
                for q in partj.parameters():
                    if p is q:
                        raise ValueError('module with duplicate parameters on distinct devices is not supported')