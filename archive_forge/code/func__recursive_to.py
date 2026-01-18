import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    """Recursively moves input to the target_device."""

    def to_map(obj):
        if isinstance(obj, (torch.Tensor, PackedSequence)):
            device = obj.data.device if isinstance(obj, PackedSequence) else obj.device
            if device == target_device:
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_device),)
            else:
                device_mod = getattr(torch, device.type, None)
                if device.type == 'cpu' or device_mod is None:
                    return (obj.to(target_device),)
                stream = _get_stream(target_device)
                with device_mod.stream(stream):
                    output = obj.to(target_device)
                with device_mod.device(target_device.index):
                    current_stream = device_mod.current_stream()
                    current_stream.wait_stream(stream)
                    if isinstance(obj, PackedSequence):
                        output.data.record_stream(current_stream)
                    else:
                        assert isinstance(output, torch.Tensor)
                        output.record_stream(current_stream)
                return (output,)
        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(to_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(to_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(to_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(to_map, obj.items()))]
        return [obj]
    try:
        res = to_map(inputs)
    finally:
        to_map = None
    return res