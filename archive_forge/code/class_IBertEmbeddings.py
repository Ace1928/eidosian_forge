import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ibert import IBertConfig
from .quant_modules import IntGELU, IntLayerNorm, IntSoftmax, QuantAct, QuantEmbedding, QuantLinear
class IBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        self.quant_mode = config.quant_mode
        self.embedding_bit = 8
        self.embedding_act_bit = 16
        self.act_bit = 8
        self.ln_input_bit = 22
        self.ln_output_bit = 32
        self.word_embeddings = QuantEmbedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, weight_bit=self.embedding_bit, quant_mode=self.quant_mode)
        self.token_type_embeddings = QuantEmbedding(config.type_vocab_size, config.hidden_size, weight_bit=self.embedding_bit, quant_mode=self.quant_mode)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.padding_idx = config.pad_token_id
        self.position_embeddings = QuantEmbedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx, weight_bit=self.embedding_bit, quant_mode=self.quant_mode)
        self.embeddings_act1 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.embeddings_act2 = QuantAct(self.embedding_act_bit, quant_mode=self.quant_mode)
        self.LayerNorm = IntLayerNorm(config.hidden_size, eps=config.layer_norm_eps, output_bit=self.ln_output_bit, quant_mode=self.quant_mode, force_dequant=config.force_dequant)
        self.output_activation = QuantAct(self.act_bit, quant_mode=self.quant_mode)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds, inputs_embeds_scaling_factor = self.word_embeddings(input_ids)
        else:
            inputs_embeds_scaling_factor = None
        token_type_embeddings, token_type_embeddings_scaling_factor = self.token_type_embeddings(token_type_ids)
        embeddings, embeddings_scaling_factor = self.embeddings_act1(inputs_embeds, inputs_embeds_scaling_factor, identity=token_type_embeddings, identity_scaling_factor=token_type_embeddings_scaling_factor)
        if self.position_embedding_type == 'absolute':
            position_embeddings, position_embeddings_scaling_factor = self.position_embeddings(position_ids)
            embeddings, embeddings_scaling_factor = self.embeddings_act1(embeddings, embeddings_scaling_factor, identity=position_embeddings, identity_scaling_factor=position_embeddings_scaling_factor)
        embeddings, embeddings_scaling_factor = self.LayerNorm(embeddings, embeddings_scaling_factor)
        embeddings = self.dropout(embeddings)
        embeddings, embeddings_scaling_factor = self.output_activation(embeddings, embeddings_scaling_factor)
        return (embeddings, embeddings_scaling_factor)

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)