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
from .configuration_roberta import RobertaConfig
@add_start_docstrings('\n    Roberta Model transformer with a sequence classification/regression head on top (a linear layer on top of the\n    pooled output) e.g. for GLUE tasks.\n    ', ROBERTA_START_DOCSTRING)
class FlaxRobertaForSequenceClassification(FlaxRobertaPreTrainedModel):
    module_class = FlaxRobertaForSequenceClassificationModule