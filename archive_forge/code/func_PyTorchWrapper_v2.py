from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
@registry.layers('PyTorchWrapper.v2')
def PyTorchWrapper_v2(pytorch_model: Any, convert_inputs: Optional[Callable]=None, convert_outputs: Optional[Callable]=None, mixed_precision: bool=False, grad_scaler: Optional[PyTorchGradScaler]=None, device: Optional['torch.device']=None) -> Model[Any, Any]:
    """Wrap a PyTorch model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch. See examples/wrap_pytorch.py

    Your PyTorch model's forward method can take arbitrary args and kwargs,
    but must return either a single tensor as output or a tuple. You may find the
    PyTorch register_forward_hook helpful if you need to adapt the output.

    The convert functions are used to map inputs and outputs to and from your
    PyTorch model. Each function should return the converted output, and a callback
    to use during the backward pass. So:

        Xtorch, get_dX = convert_inputs(X)
        Ytorch, torch_backprop = model.shims[0](Xtorch, is_train)
        Y, get_dYtorch = convert_outputs(Ytorch)

    To allow maximum flexibility, the PyTorchShim expects ArgsKwargs objects
    on the way into the forward and backward passed. The ArgsKwargs objects
    will be passed straight into the model in the forward pass, and straight
    into `torch.autograd.backward` during the backward pass.

    mixed_precision:
        Enable mixed-precision. This changes whitelisted ops to run
        in half precision for better performance and lower memory use.
    grad_scaler:
        The gradient scaler to use for mixed-precision training. If this
        argument is set to "None" and mixed precision is enabled, a gradient
        scaler with the default configuration is used.
    device:
        The PyTorch device to run the model on. When this argument is
        set to "None", the default device for the currently active Thinc
        ops is used.
    """
    if convert_inputs is None:
        convert_inputs = convert_pytorch_default_inputs
    if convert_outputs is None:
        convert_outputs = convert_pytorch_default_outputs
    return Model('pytorch', forward, attrs={'convert_inputs': convert_inputs, 'convert_outputs': convert_outputs}, shims=[PyTorchShim(pytorch_model, mixed_precision=mixed_precision, grad_scaler=grad_scaler, device=device)], dims={'nI': None, 'nO': None})