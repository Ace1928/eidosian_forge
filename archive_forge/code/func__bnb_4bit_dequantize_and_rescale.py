import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
    """
        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will
        be quantized again.
        """
    if not is_bitsandbytes_available():
        raise ImportError('Please install bitsandbytes to use this method.')
    import bitsandbytes as bnb
    dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)
    dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))
    quant_weight = bnb.nn.Params4bit(dequant_weights.to('cpu'), requires_grad=False).to(dequant_weights.device)
    setattr(target_layer, 'weight', quant_weight)