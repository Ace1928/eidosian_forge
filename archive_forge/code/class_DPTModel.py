import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
@add_start_docstrings('The bare DPT Model transformer outputting raw hidden-states without any specific head on top.', DPT_START_DOCSTRING)
class DPTModel(DPTPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        if config.is_hybrid:
            self.embeddings = DPTViTHybridEmbeddings(config)
        else:
            self.embeddings = DPTViTEmbeddings(config)
        self.encoder = DPTViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = DPTViTPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        if self.config.is_hybrid:
            return self.embeddings
        else:
            return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndIntermediateActivations, config_class=_CONFIG_FOR_DOC, modality='vision', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPoolingAndIntermediateActivations]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values, return_dict=return_dict)
        embedding_last_hidden_states = embedding_output[0] if not return_dict else embedding_output.last_hidden_states
        encoder_outputs = self.encoder(embedding_last_hidden_states, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:] + embedding_output[1:]
        return BaseModelOutputWithPoolingAndIntermediateActivations(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions, intermediate_activations=embedding_output.intermediate_activations)