import math
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig
@add_start_docstrings('The bare DistilBert Model transformer outputting raw hidden-states without any specific head on top.', FLAX_DISTILBERT_START_DOCSTRING)
class FlaxDistilBertModel(FlaxDistilBertPreTrainedModel):
    module_class = FlaxDistilBertModule