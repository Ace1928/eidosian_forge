from typing import Any, Callable, Dict, Optional, Tuple, cast
from ..compat import torch
from ..config import registry
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim
from ..types import ArgsKwargs, Floats3d, Padded
from ..util import (
@registry.layers('PyTorchRNNWrapper.v1')
def PyTorchRNNWrapper(pytorch_model: Any, convert_inputs: Optional[Callable]=None, convert_outputs: Optional[Callable]=None) -> Model[Padded, Padded]:
    """Wrap a PyTorch RNN model for use in Thinc."""
    if convert_inputs is None:
        convert_inputs = convert_rnn_inputs
    if convert_outputs is None:
        convert_outputs = convert_rnn_outputs
    return cast(Model[Padded, Padded], PyTorchWrapper(pytorch_model, convert_inputs=convert_inputs, convert_outputs=convert_outputs))