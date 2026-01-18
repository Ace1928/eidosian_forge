import inspect
import logging
import numpy as np
import torch
import torch.utils._pytree as pytree
import pennylane as qml
def _res_to_torch(r, ctx):
    """Convert results from unwrapped execution to torch."""
    if isinstance(r, dict):
        return r
    if isinstance(r, (list, tuple)):
        return type(r)((_res_to_torch(t, ctx) for t in r))
    return torch.as_tensor(r, device=ctx.torch_device)