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
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
@add_start_docstrings('Wav2Vec2 Model with a quantizer and `VQ` head on top.', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
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
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(target_features: torch.FloatTensor, negative_features: torch.FloatTensor, predicted_features: torch.FloatTensor, temperature: int=0.1):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(target_features)
        logits = logits / temperature
        return logits

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, mask_time_indices: Optional[torch.BoolTensor]=None, sampled_negative_indices: Optional[torch.BoolTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:
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
        >>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
        >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        >>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1

        >>> # compute masked indices
        >>> batch_size, raw_sequence_length = input_values.shape
        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
        >>> mask_time_indices = _compute_mask_indices(
        ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        ... )
        >>> sampled_negative_indices = _sample_negative_indices(
        ...     features_shape=(batch_size, sequence_length),
        ...     num_negatives=model.config.num_negatives,
        ...     mask_time_indices=mask_time_indices,
        ... )
        >>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
        >>> sampled_negative_indices = torch.tensor(
        ...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        ... )

        >>> with torch.no_grad():
        ...     outputs = model(input_values, mask_time_indices=mask_time_indices)

        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        >>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        >>> # show that cosine similarity is much higher than random
        >>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
        tensor(True)

        >>> # for contrastive loss training model should be put into train mode
        >>> model = model.train()
        >>> loss = model(
        ...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
        ... ).loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, mask_time_indices=mask_time_indices, return_dict=return_dict)
        transformer_features = self.project_hid(outputs[0])
        extract_features = self.dropout_features(outputs[1])
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask, add_adapter=False)
        quantized_features, codevector_perplexity = self.quantizer(extract_features, mask_time_indices=mask_time_indices)
        quantized_features = self.project_q(quantized_features)
        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape
            negative_quantized_features = quantized_features.view(-1, hidden_size)[sampled_negative_indices.long().view(-1)]
            negative_quantized_features = negative_quantized_features.view(batch_size, sequence_length, -1, hidden_size).permute(2, 0, 1, 3)
            logits = self.compute_contrastive_logits(quantized_features[None, :], negative_quantized_features, transformer_features, self.config.contrastive_logits_temperature)
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)
            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float('-inf')
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()
            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction='sum')
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = (num_codevectors - codevector_perplexity) / num_codevectors * mask_time_indices.sum()
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss
        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return Wav2Vec2ForPreTrainingOutput(loss=loss, projected_states=transformer_features, projected_quantized_states=quantized_features, codevector_perplexity=codevector_perplexity, hidden_states=outputs.hidden_states, attentions=outputs.attentions, contrastive_loss=contrastive_loss, diversity_loss=diversity_loss)