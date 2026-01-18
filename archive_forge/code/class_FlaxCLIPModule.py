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
class FlaxCLIPModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        text_config = self.config.text_config
        vision_config = self.config.vision_config
        self.projection_dim = self.config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = FlaxCLIPTextTransformer(text_config, dtype=self.dtype)
        self.vision_model = FlaxCLIPVisionTransformer(vision_config, dtype=self.dtype)
        self.visual_projection = nn.Dense(self.projection_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.02), use_bias=False)
        self.text_projection = nn.Dense(self.projection_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.02), use_bias=False)
        self.logit_scale = self.param('logit_scale', lambda _, shape: jnp.ones(shape) * self.config.logit_scale_init_value, [])

    def __call__(self, input_ids=None, pixel_values=None, attention_mask=None, position_ids=None, deterministic: bool=True, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / jnp.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / jnp.linalg.norm(text_embeds, axis=-1, keepdims=True)
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_text = jnp.matmul(text_embeds, image_embeds.T) * logit_scale
        logits_per_image = logits_per_text.T
        if not return_dict:
            return (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
        return FlaxCLIPOutput(logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)