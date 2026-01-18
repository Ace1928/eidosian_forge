from typing import Any, Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
@flax.struct.dataclass
class FlaxCLIPOutput(ModelOutput):
    """
    Args:
        logits_per_image:(`jnp.ndarray` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`jnp.ndarray` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`jnp.ndarray` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPTextModel`].
        image_embeds(`jnp.ndarray` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`FlaxCLIPVisionModel`].
        text_model_output(`FlaxBaseModelOutputWithPooling`):
            The output of the [`FlaxCLIPTextModel`].
        vision_model_output(`FlaxBaseModelOutputWithPooling`):
            The output of the [`FlaxCLIPVisionModel`].
    """
    logits_per_image: jnp.ndarray = None
    logits_per_text: jnp.ndarray = None
    text_embeds: jnp.ndarray = None
    image_embeds: jnp.ndarray = None
    text_model_output: FlaxBaseModelOutputWithPooling = None
    vision_model_output: FlaxBaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple((self[k] if k not in ['text_model_output', 'vision_model_output'] else getattr(self, k).to_tuple() for k in self.keys()))