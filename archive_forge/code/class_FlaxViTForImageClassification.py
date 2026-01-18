from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_vit import ViTConfig
@add_start_docstrings('\n    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of\n    the [CLS] token) e.g. for ImageNet.\n    ', VIT_START_DOCSTRING)
class FlaxViTForImageClassification(FlaxViTPreTrainedModel):
    module_class = FlaxViTForImageClassificationModule