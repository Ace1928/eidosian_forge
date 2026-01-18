import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
class LinearNF4(Linear4bit):
    """ Implements the NF4 data type.

        Constructs a quantization data type where each bin has equal area under a standard normal distribution N(0, 1) that
        is normalized into the range [-1, 1].

        For more information read the paper: QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)

        Implementation of the NF4 data type in bitsandbytes can be found in the `create_normal_map` function in
        the `functional.py` file: https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L236.
    """

    def __init__(self, input_features, output_features, bias=True, compute_dtype=None, compress_statistics=True, quant_storage=torch.uint8, device=None):
        """
        Args:
            input_features (`str`):
                Number of input features of the linear layer.
            output_features (`str`):
                Number of output features of the linear layer.
            bias (`bool`, defaults to `True`):
                Whether the linear class uses the bias term as well.
        """
        super().__init__(input_features, output_features, bias, compute_dtype, compress_statistics, 'nf4', quant_storage, device)