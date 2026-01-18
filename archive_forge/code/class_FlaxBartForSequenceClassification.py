import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bart import BartConfig
@add_start_docstrings('\n    Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE\n    tasks.\n    ', BART_START_DOCSTRING)
class FlaxBartForSequenceClassification(FlaxBartPreTrainedModel):
    module_class = FlaxBartForSequenceClassificationModule
    dtype = jnp.float32