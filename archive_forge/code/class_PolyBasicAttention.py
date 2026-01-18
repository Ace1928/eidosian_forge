from typing import Any, Dict, Optional, Tuple
import torch
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .biencoder import AddLabelFixedCandsTRA
from .modules import (
from .transformer import TransformerRankerAgent
class PolyBasicAttention(BasicAttention):
    """
    Override basic attention to account for edge case for polyencoder.
    """

    def __init__(self, poly_type, n_codes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly_type = poly_type
        self.n_codes = n_codes

    def forward(self, *args, **kwargs):
        """
        Forward pass.

        Account for accidental dimensionality reduction when num_codes is 1 and the
        polyencoder type is 'codes'
        """
        lhs_emb = super().forward(*args, **kwargs)
        if self.poly_type == 'codes' and self.n_codes == 1 and (len(lhs_emb.shape) == 2):
            lhs_emb = lhs_emb.unsqueeze(self.dim - 1)
        return lhs_emb