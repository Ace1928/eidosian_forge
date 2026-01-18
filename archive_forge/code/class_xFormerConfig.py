import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from xformers._deprecation_warning import deprecated_function
from xformers.components import reversible as rv
from xformers.components.residual import ResidualNormStyle, get_deepnorm_coefficients
from xformers.factory.block_configs import (
from xformers.factory.block_factory import xFormerDecoderBlock, xFormerEncoderBlock
from xformers.factory.weight_init import get_weight_init_fn, xFormerWeightInit
@dataclass(init=False)
class xFormerConfig:
    """
    The configuration structure to define a full Transformer.
    This can include a stack of encoder layers, and a stack of decoder layers.

    It is optionally possible to share the embedding weights in between
    the encoder and decoder positional encoding, as proposed for instance by
    `Using the Output Embedding to Improve Language Models`, Press et al.

    A full config example is for instance as follows:

    ::

        xformer_config = [
            {
                "reversible": False,  # Turn on to test the effect of using reversible layers
                "block_type": "encoder",
                "num_layers": LAYERS,
                "dim_model": EMB,
                "residual_norm_style": "pre",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": CONTEXT,
                    "vocab_size": VOCAB_SIZE,
                },
                "multi_head_config": {
                    "num_heads": NUM_HEADS,
                    "residual_dropout": RES_DROP,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": ATTENTION_MECHANISM_STR,
                        "dropout": ATTN_DROP,
                        "causal": True,
                        "seq_len": CONTEXT,
                    },
                },
                "feedforward_config": {
                    "name": "FusedMLP",  # Use MLP if Triton is not available
                    "dropout": MLP_DROP,
                    "activation": "gelu",
                    "hidden_layer_multiplier": MLP_MULTIPLIER,
                },
            }
        ]


    .. _`Using the Output Embedding to Improve Language Models`: https://arxiv.org/pdf/1608.05859.pdf
    """
    stack_configs: Union[List[xFormerBlockConfig], Dict[str, xFormerBlockConfig]]
    tie_embedding_weights: bool = False
    weight_init: xFormerWeightInit = xFormerWeightInit.ViT

    def __init__(self, stack_configs: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]], tie_embedding_weights: bool=False, weight_init: xFormerWeightInit=xFormerWeightInit.ViT):
        if isinstance(stack_configs, dict):
            self.stack_configs = {}
            for k, config in stack_configs.items():
                if config['block_type'] == 'encoder':
                    self.stack_configs[k] = xFormerEncoderConfig(**config)
                else:
                    self.stack_configs[k] = xFormerDecoderConfig(**config)
        else:
            self.stack_configs = []
            for config in stack_configs:
                if config['block_type'] == 'encoder':
                    self.stack_configs.append(xFormerEncoderConfig(**config))
                else:
                    self.stack_configs.append(xFormerDecoderConfig(**config))
        self.tie_embedding_weights = tie_embedding_weights
        self.weight_init = weight_init
        deprecated_function(self)