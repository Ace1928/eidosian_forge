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
def _verify_module(module: nn.Sequential) -> None:
    if not isinstance(module, nn.Sequential):
        raise TypeError('module must be nn.Sequential to be partitioned')
    named_children = list(module.named_children())
    if len(named_children) != len(module):
        raise ValueError('module with duplicate children is not supported')