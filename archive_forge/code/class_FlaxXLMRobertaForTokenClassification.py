from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xlm_roberta import XLMRobertaConfig
@add_start_docstrings('\n    XLM Roberta Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.\n    for Named-Entity-Recognition (NER) tasks.\n    ', XLM_ROBERTA_START_DOCSTRING)
class FlaxXLMRobertaForTokenClassification(FlaxXLMRobertaPreTrainedModel):
    module_class = FlaxXLMRobertaForTokenClassificationModule