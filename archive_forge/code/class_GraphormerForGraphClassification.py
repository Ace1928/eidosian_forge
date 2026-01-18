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
class GraphormerForGraphClassification(GraphormerPreTrainedModel):
    """
    This model can be used for graph-level classification or regression tasks.

    It can be trained on
    - regression (by setting config.num_classes to 1); there should be one float-type label per graph
    - one task classification (by setting config.num_classes to the number of classes); there should be one integer
      label per graph
    - binary multi-task classification (by setting config.num_classes to the number of labels); there should be a list
      of integer labels for each graph.
    """

    def __init__(self, config: GraphormerConfig):
        super().__init__(config)
        self.encoder = GraphormerModel(config)
        self.embedding_dim = config.embedding_dim
        self.num_classes = config.num_classes
        self.classifier = GraphormerDecoderHead(self.embedding_dim, self.num_classes)
        self.is_encoder_decoder = True
        self.post_init()

    def forward(self, input_nodes: torch.LongTensor, input_edges: torch.LongTensor, attn_bias: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor, spatial_pos: torch.LongTensor, attn_edge_type: torch.LongTensor, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, **unused) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos, attn_edge_type, return_dict=True)
        outputs, hidden_states = (encoder_outputs['last_hidden_state'], encoder_outputs['hidden_states'])
        head_outputs = self.classifier(outputs)
        logits = head_outputs[:, 0, :].contiguous()
        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)
            if self.num_classes == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits[mask].squeeze(), labels[mask].squeeze().float())
            elif self.num_classes > 1 and len(labels.shape) == 1:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[mask].view(-1, self.num_classes), labels[mask].view(-1))
            else:
                loss_fct = BCEWithLogitsLoss(reduction='sum')
                loss = loss_fct(logits[mask], labels[mask])
        if not return_dict:
            return tuple((x for x in [loss, logits, hidden_states] if x is not None))
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states, attentions=None)