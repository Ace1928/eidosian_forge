import math
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Module
from . import components
def _init_hubert_pretrain_model(module):
    if isinstance(module, components.ConvLayerBlock):
        torch.nn.init.kaiming_normal_(module.conv.weight)
    elif isinstance(module, components.ConvolutionalPositionalEmbedding):
        std = math.sqrt(4.0 / (module.embed_dim * module.kernel_size))
        torch.nn.init.normal_(module.conv.weight, mean=0.0, std=std)
        torch.nn.init.constant_(module.conv.bias, 0.0)
    elif isinstance(module, components.SelfAttention):
        torch.nn.init.xavier_uniform_(module.k_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.v_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.q_proj.weight, gain=1 / math.sqrt(2))
        torch.nn.init.xavier_uniform_(module.out_proj.weight)
        torch.nn.init.constant_(module.out_proj.bias, 0.0)
    elif isinstance(module, components.Transformer):
        module.apply(components._init_transformer_params)
    else:
        pass