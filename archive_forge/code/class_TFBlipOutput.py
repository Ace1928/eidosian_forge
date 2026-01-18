from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
@dataclass
class TFBlipOutput(ModelOutput):
    """
    Args:
        loss (`tf.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`tf.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`tf.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`tf.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """
    loss: tf.Tensor | None = None
    logits_per_image: tf.Tensor = None
    logits_per_text: tf.Tensor = None
    text_embeds: tf.Tensor = None
    image_embeds: tf.Tensor = None
    text_model_output: TFBaseModelOutputWithPooling = None
    vision_model_output: TFBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))