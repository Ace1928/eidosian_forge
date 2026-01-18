import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput, Wav2Vec2BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_unispeech import UniSpeechConfig
@add_start_docstrings('The bare UniSpeech Model transformer outputting raw hidden-states without any specific head on top.', UNISPEECH_START_DOCSTRING)
class UniSpeechModel(UniSpeechPreTrainedModel):

    def __init__(self, config: UniSpeechConfig):
        super().__init__(config)
        self.config = config
        self.feature_extractor = UniSpeechFeatureEncoder(config)
        self.feature_projection = UniSpeechFeatureProjection(config)
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        if config.do_stable_layer_norm:
            self.encoder = UniSpeechEncoderStableLayerNorm(config)
        else:
            self.encoder = UniSpeechEncoder(config)
        self.post_init()

    def _mask_hidden_states(self, hidden_states: torch.FloatTensor, mask_time_indices: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """
        if not getattr(self.config, 'apply_spec_augment', True):
            return hidden_states
        batch_size, sequence_length, hidden_size = hidden_states.size()
        if mask_time_indices is not None:
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=self.config.mask_time_prob, mask_length=self.config.mask_time_length, attention_mask=attention_mask, min_masks=self.config.mask_time_min_masks)
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        if self.config.mask_feature_prob > 0 and self.training:
            mask_feature_indices = _compute_mask_indices((batch_size, hidden_size), mask_prob=self.config.mask_feature_prob, mask_length=self.config.mask_feature_length, min_masks=self.config.mask_feature_min_masks)
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0
        return hidden_states

    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=Wav2Vec2BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality='audio', expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, mask_time_indices: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)
        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return Wav2Vec2BaseModelOutput(last_hidden_state=hidden_states, extract_features=extract_features, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)