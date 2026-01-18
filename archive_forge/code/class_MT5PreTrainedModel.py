import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_mt5 import MT5Config
class MT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MT5Config
    load_tf_weights = load_tf_weights_in_mt5
    base_model_prefix = 'transformer'
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['MT5Block']
    _keep_in_fp32_modules = ['wo']

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {'decoder_input_ids': input_ids, 'input_ids': input_ids, 'decoder_attention_mask': input_mask}
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, MT5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (MT5Model, MT5ForConditionalGeneration, MT5EncoderModel, MT5ForQuestionAnswering)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'lm_head') and (not self.config.tie_word_embeddings):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'qa_outputs'):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
                module.qa_outputs.bias.data.zero_()
        elif isinstance(module, MT5ForTokenClassification):
            if hasattr(module, 'classifier'):
                module.classifier.weight.data.normal_(mean=0.0, std=factor * 1.0)
                module.classifier.bias.data.zero_()
        elif isinstance(module, MT5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.dense, 'bias') and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, MT5DenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MT5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, MT5Attention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In MT5 it is usually set to the pad_token_id. See MT5 docs for more information.')
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError('self.model.config.pad_token_id has to be defined.')
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids