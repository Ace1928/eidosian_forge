import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn import functional as F
def _get_linear_layer(in_dim: int, out_dim: int, bias: bool=True, w_init_gain: str='linear') -> torch.nn.Linear:
    """Linear layer with xavier uniform initialization.

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. (Default: ``True``)
        w_init_gain (str, optional): Parameter passed to ``torch.nn.init.calculate_gain``
            for setting the gain parameter of ``xavier_uniform_``. (Default: ``linear``)

    Returns:
        (torch.nn.Linear): The corresponding linear layer.
    """
    linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
    torch.nn.init.xavier_uniform_(linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    return linear