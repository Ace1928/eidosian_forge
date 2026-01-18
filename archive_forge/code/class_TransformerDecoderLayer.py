import math
import torch
import torch.nn as nn
from fairscale.nn.moe.moe_layer import MOELayer
from fairscale.nn.moe.top2gate import Top2Gate
class TransformerDecoderLayer(TransformerEncoderLayer):
    """TransformerDecoder layer which inherits from TransformerEncoderLayer."""

    def __init__(self, ninp, nhead, nhid, dropout, is_moe=False, num_local_experts=1):
        super().__init__(ninp, nhead, nhid, dropout, is_moe=is_moe, num_local_experts=num_local_experts)
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        return super().forward(src, self.src_mask)