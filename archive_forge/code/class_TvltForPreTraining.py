import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
@add_start_docstrings('The TVLT Model transformer with the decoder on top for self-supervised pre-training.', TVLT_START_DOCSTRING)
class TvltForPreTraining(TvltPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.task_matching = config.task_matching
        self.task_mae = config.task_mae
        if not (self.task_matching or self.task_mae):
            raise ValueError('Must set at least one of matching task and MAE task to true')
        self.tvlt = TvltModel(config)
        if self.task_matching:
            self.matching_head = TvltMatchingHead(config)
        if self.task_mae:
            self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
            self.pixel_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
            self.audio_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
            self.decoder = TvltDecoder(config)
            decoder_hidden_size = config.decoder_hidden_size
            num_frames = config.num_frames
            num_patches_per_image = self.tvlt.pixel_embeddings.num_patches_per_image
            self.decoder_pixel_pos_embed = nn.Parameter(torch.zeros(1, num_patches_per_image, decoder_hidden_size))
            self.decoder_temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, decoder_hidden_size))
            self.decoder_pixel_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
            num_audio_patches = self.tvlt.audio_embeddings.num_patches
            num_freq_patches = config.frequency_length // config.audio_patch_size[1]
            self.decoder_audio_pos_embed = nn.Parameter(torch.zeros(1, num_audio_patches // num_freq_patches, decoder_hidden_size))
            self.decoder_freq_embed = nn.Parameter(torch.zeros(1, num_freq_patches, decoder_hidden_size))
            self.decoder_audio_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
            pixel_mae_output_dim = self.config.image_patch_size[0] ** 2 * self.config.num_image_channels
            self.pixel_mae_head = TvltMAEHead(config, pixel_mae_output_dim)
            audio_mae_output_dim = self.config.audio_patch_size[0] * self.config.audio_patch_size[1] * self.config.num_audio_channels
            self.audio_mae_head = TvltMAEHead(config, audio_mae_output_dim)
            self.num_frames = num_frames
            self.num_patches_per_image = num_patches_per_image
            self.num_freq_patches = num_freq_patches
            self.image_patch_size = config.image_patch_size
            self.audio_patch_size = config.audio_patch_size
        self.post_init()

    def patchify_pixel(self, pixel_values):
        """
        pixel_values: [batch_size, num_frames, 3, height, width]
        """
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        num_patches_height = pixel_values.shape[3] // self.image_patch_size[0]
        num_patches_width = pixel_values.shape[4] // self.image_patch_size[1]
        patchified_pixel_values = pixel_values.reshape(shape=(batch_size, num_frames, num_channels, num_patches_height, self.image_patch_size[0], num_patches_width, self.image_patch_size[1]))
        patchified_pixel_values = torch.einsum('ntchpwq->nthwpqc', patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(shape=(batch_size, num_patches_height * num_patches_width * num_frames, self.image_patch_size[0] * self.image_patch_size[1] * num_channels))
        return patchified_pixel_values

    def patchify_audio(self, audio_values):
        """
        audio_values: [batch_size, 1, height, width]
        """
        batch_size, num_channels, height, width = audio_values.shape
        num_patches_height = height // self.audio_patch_size[0]
        num_patches_width = width // self.audio_patch_size[1]
        patchified_audio_values = audio_values.reshape(shape=(batch_size, num_channels, num_patches_height, self.audio_patch_size[0], num_patches_width, self.audio_patch_size[1]))
        patchified_audio_values = torch.einsum('nchpwq->nhwpqc', patchified_audio_values)
        patchified_audio_values = patchified_audio_values.reshape(shape=(batch_size, num_patches_height * num_patches_width, self.audio_patch_size[0] * self.audio_patch_size[1] * num_channels))
        return patchified_audio_values

    def pixel_mae_loss(self, pixel_values, pixel_predictions, mask):
        patchified_pixel_values = self.patchify_pixel(pixel_values)
        loss = (pixel_predictions - patchified_pixel_values) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def audio_mae_loss(self, audio_values, audio_predictions, mask):
        patchified_audio_values = self.patchify_audio(audio_values)
        loss = (audio_predictions - patchified_audio_values) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def concatenate_mask(self, mask_token, sequence, ids_restore):
        batch_size, seq_length, dim = sequence.shape
        mask_tokens = mask_token.repeat(batch_size, ids_restore.shape[1] - seq_length, 1)
        padded_sequence = torch.cat([sequence, mask_tokens], dim=1)
        padded_sequence = torch.gather(padded_sequence, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim))
        return padded_sequence

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, audio_values: torch.FloatTensor, pixel_mask: Optional[torch.FloatTensor]=None, audio_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, pixel_values_mixed: Optional[torch.FloatTensor]=None, pixel_mask_mixed: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], TvltForPreTrainingOutput]:
        """
        pixel_values_mixed (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values that mix positive and negative samples in Tvlt vision-audio matching. Audio values can be
            obtained using [`TvltProcessor`]. See [`TvltProcessor.__call__`] for details.

        pixel_mask_mixed (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel masks of pixel_values_mixed. Pixel values mixed can be obtained using [`TvltProcessor`]. See
            [`TvltProcessor.__call__`] for details.

        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the vision audio matching loss. Indices should be in `[0, 1]`. num_labels has to be 1.

        Return:

        Examples:

        ```python
        >>> from transformers import TvltProcessor, TvltForPreTraining
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> images_mixed = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))
        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltForPreTraining.from_pretrained("ZinengTang/tvlt-base")
        >>> input_dict = processor(
        ...     images, audio, images_mixed, sampling_rate=44100, mask_pixel=True, mask_audio=True, return_tensors="pt"
        ... )

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        total_loss = 0.0
        if self.task_matching:
            if labels is None:
                raise ValueError('Matching task requires labels')
            if pixel_values_mixed is None:
                raise ValueError('Matching task requires pixel_values_mixed')
            outputs = self.tvlt(pixel_values_mixed, audio_values, pixel_mask=pixel_mask_mixed, audio_mask=audio_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            sequence_output = outputs[0]
            matching_logits = self.matching_head(sequence_output)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(matching_logits.view(-1), labels.view(-1))
            total_loss += loss
        pixel_logits = None
        audio_logits = None
        if self.task_mae and self.training:
            outputs = self.tvlt(pixel_values, audio_values, pixel_mask=pixel_mask, audio_mask=audio_mask, mask_pixel=True, mask_audio=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
            pixel_sequence_output = outputs.last_pixel_hidden_state if return_dict else outputs[1]
            audio_sequence_output = outputs.last_audio_hidden_state if return_dict else outputs[2]
            pixel_label_masks = outputs.pixel_label_masks if return_dict else outputs[3]
            audio_label_masks = outputs.audio_label_masks if return_dict else outputs[4]
            pixel_ids_restore = outputs.pixel_ids_restore if return_dict else outputs[5]
            audio_ids_restore = outputs.audio_ids_restore if return_dict else outputs[6]
            pixel_decoder_input = self.encoder_to_decoder(pixel_sequence_output)
            audio_decoder_input = self.encoder_to_decoder(audio_sequence_output)
            num_frames = pixel_values.size(1)
            pixel_decoder_input = self.concatenate_mask(self.pixel_mask_token, pixel_decoder_input, pixel_ids_restore)
            pixel_decoder_input = pixel_decoder_input + self.decoder_pixel_pos_embed.repeat(1, num_frames, 1)
            pixel_decoder_input = pixel_decoder_input + torch.repeat_interleave(self.decoder_temporal_embed[:, :num_frames], self.num_patches_per_image, dim=1)
            pixel_decoder_input = pixel_decoder_input + self.decoder_pixel_type_embed
            pixel_decoder_outputs = self.decoder(pixel_decoder_input)
            pixel_logits = self.pixel_mae_head(pixel_decoder_outputs.logits)
            audio_decoder_input = self.concatenate_mask(self.audio_mask_token, audio_decoder_input, audio_ids_restore)
            num_time_patches = audio_decoder_input.size(1) // self.num_freq_patches
            audio_decoder_input = audio_decoder_input + self.decoder_freq_embed.repeat(1, num_time_patches, 1)
            audio_decoder_input = audio_decoder_input + torch.repeat_interleave(self.decoder_audio_pos_embed[:, :num_time_patches], self.num_freq_patches, dim=1)
            audio_decoder_input = audio_decoder_input + self.decoder_audio_type_embed
            audio_decoder_outputs = self.decoder(audio_decoder_input)
            audio_logits = self.audio_mae_head(audio_decoder_outputs.logits)
            loss = self.pixel_mae_loss(pixel_values, pixel_logits, pixel_label_masks) + self.audio_mae_loss(audio_values, audio_logits, audio_label_masks)
            total_loss += loss
        if not return_dict:
            output = (matching_logits, pixel_logits, audio_logits) + outputs[7:]
            return (total_loss,) + output if loss is not None else output
        return TvltForPreTrainingOutput(loss=total_loss, matching_logits=matching_logits, pixel_logits=pixel_logits, audio_logits=audio_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)