from dataclasses import dataclass
import torch
from xformers.components.attention import Attention, AttentionConfig, register_attention
@register_attention('lambda', LambdaLayerConfig)
class LambdaLayer(Attention):

    def __init__(self, dropout: float, seq_len: int, dim_head: int, *_, **__):
        """
        Attention approximation using Lambda layers, from
        "Lambda networks: modeling long-range interactions without attention.", Bello, I. (2021).
        """
        super().__init__()
        self.rel_pos_emb = torch.nn.Parameter(torch.randn(2 * seq_len - 1, int(dim_head)))
        self.rel_pos = calc_rel_pos(seq_len)
        self.attn_drop = torch.nn.Dropout(dropout, inplace=True)
        self.requires_same_k_q_dimensions = True
        self.supports_attention_mask = False
        self.supports_key_padding_mask = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs):
        """..NOTE: We're reusing the einsum notation suggested by the paper, changed in that
        heads are folded in the batch dimension"""
        content_lambda = torch.einsum('bnk,bnv->bkv', torch.softmax(k, dim=-1), v)
        content_output = torch.einsum('bnk,bkv->bnv', q, content_lambda)
        rel_pos_emb = self.rel_pos_emb[self.rel_pos]
        seq_len = q.shape[1]
        rel_pos_emb = rel_pos_emb[:seq_len, :seq_len, :]
        position_lambdas = torch.einsum('mnk,bnv->bnkv', rel_pos_emb, v)
        position_output = (q.unsqueeze(2) @ position_lambdas).squeeze()
        att = content_output + position_output
        att = self.attn_drop(att)
        return att