from contextlib import contextmanager
from dataclasses import dataclass
import functools
import threading
from typing import Any, Dict, Generator, Optional, Tuple
import weakref
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from fairscale.internal.containers import pack_kwargs, split_non_tensors, unpack_kwargs, unpack_non_tensors
from .checkpoint_utils import patch_batchnorm
def _checkpointed_forward(original_forward: Any, weak_self: Any, offload_to_cpu: bool, *args: Any, **kwargs: Any) -> Any:
    module = weak_self()
    if not torch.is_grad_enabled() or thread_local.is_checkpointing_disabled:
        return original_forward(module, *args, **kwargs)
    args = (module,) + args
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {'offload': offload_to_cpu}
    output = CheckpointFunction.apply(torch.tensor([], requires_grad=True), original_forward, parent_ctx_dict, kwarg_keys, *flat_args)
    output_requires_grad = parent_ctx_dict['output_requires_grad']
    if not isinstance(output, torch.Tensor):
        output = [x.detach() if not output_requires_grad else x for x in output]
        packed_non_tensor_outputs = parent_ctx_dict['packed_non_tensor_outputs']
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
    elif not output_requires_grad:
        output = output.detach()
    return output