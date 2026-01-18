import numpy as np
from typing import Union, Tuple, Any, List
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
class AppendBiasLayer(nn.Module):
    """Simple bias appending layer for free_log_std."""

    def __init__(self, num_bias_vars: int):
        super().__init__()
        self.log_std = torch.nn.Parameter(torch.as_tensor([0.0] * num_bias_vars))
        self.register_parameter('log_std', self.log_std)

    def forward(self, x: TensorType) -> TensorType:
        out = torch.cat([x, self.log_std.unsqueeze(0).repeat([len(x), 1])], axis=1)
        return out