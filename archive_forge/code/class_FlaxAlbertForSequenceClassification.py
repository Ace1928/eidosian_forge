from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig
@add_start_docstrings('\n    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled\n    output) e.g. for GLUE tasks.\n    ', ALBERT_START_DOCSTRING)
class FlaxAlbertForSequenceClassification(FlaxAlbertPreTrainedModel):
    module_class = FlaxAlbertForSequenceClassificationModule