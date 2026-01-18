import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
class GraphormerGraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.multi_hop_max_dist = config.multi_hop_max_dist
        self.edge_encoder = nn.Embedding(config.num_edges + 1, config.num_attention_heads, padding_idx=0)
        self.edge_type = config.edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(config.num_edge_dis * config.num_attention_heads * config.num_attention_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(config.num_spatial, config.num_attention_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, config.num_attention_heads)

    def forward(self, input_nodes: torch.LongTensor, attn_bias: torch.Tensor, spatial_pos: torch.LongTensor, input_edges: torch.LongTensor, attn_edge_type: torch.LongTensor) -> torch.Tensor:
        n_graph, n_node = input_nodes.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        if self.edge_type == 'multi_hop':
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                input_edges = input_edges[:, :, :, :self.multi_hop_max_dist, :]
            input_edges = self.edge_encoder(input_edges).mean(-2)
            max_dist = input_edges.size(-2)
            edge_input_flat = input_edges.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads)[:max_dist, :, :])
            input_edges = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            input_edges = (input_edges.sum(-2) / spatial_pos_.float().unsqueeze(-1)).permute(0, 3, 1, 2)
        else:
            input_edges = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + input_edges
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        return graph_attn_bias