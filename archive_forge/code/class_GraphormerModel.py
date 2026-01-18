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
class GraphormerModel(GraphormerPreTrainedModel):
    """The Graphormer model is a graph-encoder model.

    It goes from a graph to its representation. If you want to use the model for a downstream classification task, use
    GraphormerForGraphClassification instead. For any other downstream task, feel free to add a new class, or combine
    this model with a downstream model of your choice, following the example in GraphormerForGraphClassification.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__(config)
        self.max_nodes = config.max_nodes
        self.graph_encoder = GraphormerGraphEncoder(config)
        self.share_input_output_embed = config.share_input_output_embed
        self.lm_output_learned_bias = None
        self.load_softmax = not getattr(config, 'remove_head', False)
        self.lm_head_transform_weight = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.activation_fn = ACT2FN[config.activation_fn]
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.post_init()

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, perturb: Optional[torch.FloatTensor]=None, masked_tokens: None=None, return_dict: Optional[bool]=None, **unused) -> Union[Tuple[torch.LongTensor], BaseModelOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inner_states, graph_rep = self.graph_encoder(input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, perturb=perturb)
        input_nodes = inner_states[-1].transpose(0, 1)
        if masked_tokens is not None:
            raise NotImplementedError
        input_nodes = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(input_nodes)))
        if self.share_input_output_embed and hasattr(self.graph_encoder.embed_tokens, 'weight'):
            input_nodes = torch.nn.functional.linear(input_nodes, self.graph_encoder.embed_tokens.weight)
        if not return_dict:
            return tuple((x for x in [input_nodes, inner_states] if x is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=input_nodes, hidden_states=inner_states)

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes