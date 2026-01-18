from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
from xformers.utils import generate_matching_config
@dataclass(init=False)
class xFormerEncoderConfig(xFormerBlockConfig):
    """
    The configuration structure for an encoder block
    """
    multi_head_config: Dict[str, Any]
    use_triton: bool
    simplicial_embeddings: Optional[Dict[str, Any]]
    patch_embedding_config: Optional[Dict[str, Any]]

    def __init__(self, dim_model: int, feedforward_config: Dict[str, Any], multi_head_config: Dict[str, Any], position_encoding_config: Optional[Dict[str, Any]]=None, residual_norm_style: str='post', normalization: NormalizationType=NormalizationType.LayerNorm, use_triton: bool=True, simplicial_embeddings: Optional[Dict[str, Any]]=None, patch_embedding_config: Optional[Dict[str, Any]]=None, **kwargs):
        try:
            if 'dim_model' not in multi_head_config.keys():
                multi_head_config['dim_model'] = dim_model
            if 'dim_model' not in feedforward_config.keys():
                feedforward_config['dim_model'] = dim_model
            if position_encoding_config is not None and 'dim_model' not in position_encoding_config.keys():
                position_encoding_config['dim_model'] = dim_model
            if patch_embedding_config is not None and 'out_channels' not in patch_embedding_config.keys():
                patch_embedding_config['out_channels'] = dim_model
        except AttributeError:
            pass
        if 'block_type' in kwargs:
            assert kwargs['block_type'] == 'encoder'
        kwargs['block_type'] = BlockType('encoder')
        super().__init__(dim_model=dim_model, feedforward_config=feedforward_config, position_encoding_config=position_encoding_config, residual_norm_style=ResidualNormStyle(residual_norm_style), normalization=NormalizationType(normalization), **kwargs)
        self.multi_head_config = multi_head_config
        self.use_triton = use_triton
        self.simplicial_embeddings = simplicial_embeddings
        self.patch_embedding_config = patch_embedding_config