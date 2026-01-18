import numpy as np
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch, TensorType
def _f_epsilon(self, x: TensorType) -> TensorType:
    return torch.sign(x) * torch.pow(torch.abs(x), 0.5)