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
@add_start_docstrings('UniSpeech Model with a vector-quantization module and ctc loss for pre-training.', UNISPEECH_START_DOCSTRING)
class UniSpeechForPreTraining(UniSpeechPreTrainedModel):

    def __init__(self, config: UniSpeechConfig):
        super().__init__(config)
        self.unispeech = UniSpeechModel(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        self.quantizer = UniSpeechGumbelVectorQuantizer(config)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        self.project_hid = nn.Linear(config.proj_codevector_dim, config.hidden_size)
        self.ctc_proj = nn.Linear(config.hidden_size, config.num_ctc_classes)
        self.dropout = nn.Dropout(config.final_dropout)
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn('The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.', FutureWarning)
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(target_features: torch.FloatTensor, negative_features: torch.FloatTensor, predicted_features: torch.FloatTensor, temperature: int=1):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        logits = logits.type_as(target_features)
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(UNISPEECH_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UniSpeechForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, UniSpeechForPreTrainingOutput]:
        """
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        sampled_negative_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, UniSpeechForPreTraining

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-large-1500h-cv")
        >>> model = UniSpeechForPreTraining.from_pretrained("microsoft/unispeech-large-1500h-cv")
        >>> # TODO: Add full pretraining example
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.unispeech(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        transformer_features = outputs[0]
        extract_features = self.dropout_features(outputs[1])
        quantized_features, codevector_perplexity = self.quantizer(extract_features)
        quantized_features = self.project_q(quantized_features)
        quantized_features = self.project_hid(quantized_features)
        prob_replace_matrix = torch.empty(transformer_features.size(0), transformer_features.size(1)).fill_(self.config.replace_prob)
        prob_replace_matrix = prob_replace_matrix.transpose(0, 1)
        sampled_replace_matrix = torch.bernoulli(prob_replace_matrix).bool().to(transformer_features.device)
        sampled_replace_matrix = sampled_replace_matrix.transpose(0, 1)
        sampled_replace_matrix = sampled_replace_matrix.unsqueeze(-1)
        logits = transformer_features.masked_fill(sampled_replace_matrix, 0.0) + quantized_features.masked_fill(~sampled_replace_matrix, 0.0)
        logits = self.dropout(logits)
        logits = self.ctc_proj(logits)
        loss = None
        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return UniSpeechForPreTrainingOutput(loss=loss, projected_states=transformer_features, projected_quantized_states=quantized_features, codevector_perplexity=codevector_perplexity, hidden_states=outputs.hidden_states, attentions=outputs.attentions)