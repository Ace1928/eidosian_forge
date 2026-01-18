import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
@dataclass
class FlavaForPreTrainingOutput(ModelOutput):
    """
    Output from FlavaForPreTraining containing embeddings, and outputs from individual encoders.

    Note that `image_embeddings` and `text_embeddings` returned are similar to pooled output returned from a
    transformer. If you want embeddings for contrastive loss or retrieval use a FLAVA model's `image_projection` and
    `text_projection` layers on `image_embeddings` and `text_embeddings` respectively.

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `return_loss` is True):
            Total loss calculated for this model.
        loss_info (`FlavaLosses`):
            Detailed info for FLAVA Pretraining losses. Check `FlavaLosses` class description for the information on
            the keys.
        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present):
            The image embeddings which are basically the pooled output of [`FlavaImageModel`].
        image_output (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present):
            The output of the [`FlavaImageModel`].
        text_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` are present):
            The text embeddings which are basically the pooled output of [`FlavaTextModel`].
        text_output (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids` are present):
            The output of the [`FlavaTextModel`].
        multimodal_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`):
            The multimodal embeddings which are basically the pooled output of [`FlavaTextModel`].
        multimodal_output (`BaseModelOutputWithPooling`, returned when `input_ids` and `pixel_values` are present and `skip_unmasked_multimodal_encoder` is `None` or `False`):
            The output of the [`FlavaMultimodalModel`].

        image_masked_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `pixel_values` are present):
            The image embeddings which are basically the pooled output of [`FlavaImageModel`]. Uses `bool_masked_pos`
            to create masked images.
        image_masked_output (`BaseModelOutputWithPooling`, *optional*, returned when `pixel_values` are present):
            The output of the [`FlavaImageModel`]. Uses `bool_masked_pos` to create masked images.
        text_masked_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids_masked` are present):
            The text embeddings which are basically the pooled output of [`FlavaTextModel`].
        text_masked_output (`BaseModelOutputWithPooling`, *optional*, returned when `input_ids_masked` are present):
            The output of the [`FlavaTextModel`].
        multimodal_masked_embeddings (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when `input_ids` and `pixel_values` are present):
            The multimodal embeddings which are basically the pooled output of [`FlavaTextModel`].
        multimodal_masked_output (`BaseModelOutputWithPooling`, returned when `input_ids_masked` and `pixel_values` are present):
            The output of the [`FlavaMultimodalModel`].

        mim_logits (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape `(total_masked_patches, image_vocab_size)` , *optional*, returned when `pixel_values` are present and `input_ids_masked` are not):
                The logits for MIM unimodal loss. Uses `book_masked_pos` to get masked patches. The flattened output is
                returned when `bool_masked_pos` has some of the patches masked.
        mlm_logits (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(total_masked_seq_length, text_vocab_size)`, *optional*, returned when `input_ids_masked` are present and `pixel_values` are not):
                The logits for MLM unimodal loss. The flattened output is returned when `input_ids_masked` has some of
                the tokens masked.
        itm_logits (`torch.FloatTensor` of shape `(batch_size, 2)`, *optional*, returned when `input_ids_masked` and `pixel_values` are present):
                The logits for ITM loss. Note that ITM loss is calculated on masked pairs in FLAVA.
        mmm_image_logits (`torch.FloatTensor` of shape `(batch_size, num_image_patches, image_vocab_size)` or of shape`(total_masked_patches, image_vocab_size)`, *optional*, returned when `pixel_values` and `input_ids_masked` are present):
                The logits for MMM image multimodal loss. Uses `book_masked_pos` to get masked patches. The flattened
                output is returned when `bool_masked_pos` has some of the patches masked.
        mmm_text_logits (`torch.FloatTensor` of shape `(batch_size, text_seq_length, text_vocab_size)` or of shape `(`(total_masked_seq_length, text_vocab_size)`), *optional*, returned when `pixel_values` and `input_ids_masked` are present):
                The logits for MMM text multimodal loss. The flattened output is returned when `input_ids_masked` has
                some of the tokens masked.
        contrastive_logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeddings` and `text_embeddings` but passed through FLAVA's
            `image_projection` and `text_projection` layers respectively. This represents the image-text similarity
            scores. This is calculated on unmasked images and texts.
        contrastive_logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeddings` and `image_embeddings` but passed through FLAVA's
            `text_projection` and `image_projection` layers respectively. This is calculated on unmasked images and
            texts.
    """
    loss: Optional[torch.FloatTensor] = None
    loss_info: FlavaLosses = None
    image_embeddings: Optional[torch.FloatTensor] = None
    image_output: Optional[BaseModelOutputWithPooling] = None
    text_embeddings: Optional[torch.FloatTensor] = None
    text_output: Optional[BaseModelOutputWithPooling] = None
    multimodal_embeddings: Optional[torch.FloatTensor] = None
    multimodal_output: Optional[BaseModelOutputWithPooling] = None
    image_masked_embeddings: Optional[torch.FloatTensor] = None
    image_masked_output: Optional[BaseModelOutputWithPooling] = None
    text_masked_embeddings: Optional[torch.FloatTensor] = None
    text_masked_output: Optional[BaseModelOutputWithPooling] = None
    multimodal_masked_embeddings: Optional[torch.FloatTensor] = None
    multimodal_masked_output: Optional[BaseModelOutputWithPooling] = None
    mim_logits: Optional[torch.FloatTensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    itm_logits: Optional[torch.FloatTensor] = None
    contrastive_logits_per_image: Optional[torch.FloatTensor] = None
    contrastive_logits_per_text: Optional[torch.FloatTensor] = None
    mmm_image_logits: Optional[torch.FloatTensor] = None
    mmm_text_logits: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        transformer_outputs = ['text_output', 'image_output', 'multimodal_output', 'text_masked_output', 'image_masked_output', 'multimodal_masked_output']
        return tuple((self[k] if k not in transformer_outputs else getattr(self, k).to_tuple() for k in self.keys()))