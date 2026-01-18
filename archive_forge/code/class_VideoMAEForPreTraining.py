import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .configuration_videomae import VideoMAEConfig
@add_start_docstrings('The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.', VIDEOMAE_START_DOCSTRING)
class VideoMAEForPreTraining(VideoMAEPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.videomae = VideoMAEModel(config)
        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.position_embeddings = get_sinusoid_encoding_table(self.videomae.embeddings.num_patches, config.decoder_hidden_size)
        self.decoder = VideoMAEDecoder(config, num_patches=self.videomae.embeddings.num_patches)
        self.post_init()

    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VideoMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: torch.FloatTensor, bool_masked_pos: torch.BoolTensor, head_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, VideoMAEForPreTrainingOutput]:
        """
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
            batch must have the same number of masked patches. Sequence length is `(num_frames // tubelet_size) *
            (image_size // patch_size) ** 2`.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, VideoMAEForPreTraining
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 16
        >>> video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

        >>> pixel_values = image_processor(video, return_tensors="pt").pixel_values

        >>> num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
        >>> seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
        >>> bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.videomae(pixel_values, bool_masked_pos=bool_masked_pos, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.encoder_to_decoder(sequence_output)
        batch_size, seq_len, num_channels = sequence_output.shape
        if bool_masked_pos is None:
            raise ValueError('One must provided a boolean mask ')
        expanded_position_embeddings = self.position_embeddings.expand(batch_size, -1, -1).type_as(pixel_values)
        expanded_position_embeddings = expanded_position_embeddings.to(pixel_values.device).clone().detach()
        pos_emb_visible = expanded_position_embeddings[~bool_masked_pos].reshape(batch_size, -1, num_channels)
        pos_emb_mask = expanded_position_embeddings[bool_masked_pos].reshape(batch_size, -1, num_channels)
        x_full = torch.cat([sequence_output + pos_emb_visible, self.mask_token + pos_emb_mask], dim=1)
        decoder_outputs = self.decoder(x_full, pos_emb_mask.shape[1])
        logits = decoder_outputs.logits
        loss = None
        with torch.no_grad():
            if self.config.num_channels != 3:
                frames = pixel_values
            else:
                device = pixel_values.device
                dtype = pixel_values.dtype
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device=device, dtype=dtype)[None, None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device=device, dtype=dtype)[None, None, :, None, None]
                frames = pixel_values * std + mean
            batch_size, time, num_channels, height, width = frames.shape
            tubelet_size, patch_size = (self.config.tubelet_size, self.config.patch_size)
            if self.config.norm_pix_loss:
                frames = frames.view(batch_size, time // tubelet_size, tubelet_size, num_channels, height // patch_size, patch_size, width // patch_size, patch_size)
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                frames = frames.view(batch_size, time // tubelet_size * height // patch_size * width // patch_size, tubelet_size * patch_size * patch_size, num_channels)
                frames_norm = (frames - frames.mean(dim=-2, keepdim=True)) / (frames.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-06)
                videos_patch = frames_norm.view(batch_size, time // tubelet_size * height // patch_size * width // patch_size, tubelet_size * patch_size * patch_size * num_channels)
            else:
                if self.config.num_channels != 3:
                    raise ValueError("Can't unnormalize non-RGB images. Consider setting config.norm_pix_loss to False.")
                frames = frames.view(batch_size, time // tubelet_size, tubelet_size, num_channels, height // patch_size, patch_size, width // patch_size, patch_size)
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                videos_patch = frames.view(batch_size, time // tubelet_size * height // patch_size * width // patch_size, tubelet_size * patch_size * patch_size * num_channels)
            batch_size, _, num_channels = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(batch_size, -1, num_channels)
        loss_fct = MSELoss()
        loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return VideoMAEForPreTrainingOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)