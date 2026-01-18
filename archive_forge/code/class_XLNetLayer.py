import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_xlnet import XLNetConfig
class XLNetLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(self, output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=None, target_mapping=None, head_mask=None, output_attentions=False):
        outputs = self.rel_attn(output_h, output_g, attn_mask_h, attn_mask_g, r, seg_mat, mems=mems, target_mapping=target_mapping, head_mask=head_mask, output_attentions=output_attentions)
        output_h, output_g = outputs[:2]
        if output_g is not None:
            output_g = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_g)
        output_h = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_h)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs

    def ff_chunk(self, output_x):
        output_x = self.ff(output_x)
        return output_x