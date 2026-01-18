import numpy as np
from typing import Union, Tuple, Any, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self, in_size: int, out_size: int, initializer: Any=None, activation_fn: Any=None, use_bias: bool=True, bias_init: float=0.0):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size: Input size for FC Layer
            out_size: Output size for FC Layer
            initializer: Initializer function for FC layer weights
            activation_fn: Activation function at the end of layer
            use_bias: Whether to add bias weights or not
            bias_init: Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, 'torch')
        if activation_fn is not None:
            layers.append(activation_fn())
        self._model = nn.Sequential(*layers)

    def forward(self, x: TensorType) -> TensorType:
        return self._model(x)