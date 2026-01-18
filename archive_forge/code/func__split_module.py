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
def _split_module(modules: nn.Sequential) -> Tuple[List[nn.Sequential], List[torch.device]]:
    partitions = []
    devices = []
    current_partition = []
    current_device = None
    for name, module in modules.named_children():
        if isinstance(module, WithDevice):
            device = module.device
            module = module.module
            module.to(device)
        else:
            device = _retrieve_device(module)
        if current_device is not None and (current_device != device or device.type == 'cpu'):
            partitions.append(_assemble_partition(current_partition))
            devices.append(current_device)
            current_partition = []
        current_device = device
        current_partition.append(module)
    if current_device is not None:
        partitions.append(_assemble_partition(current_partition))
        devices.append(current_device)
    partitions = cast(List[nn.Sequential], nn.ModuleList(partitions))
    return (partitions, devices)