from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
@add_start_docstrings("Beit Model transformer with a 'language' modeling head on top (to predict visual tokens).", BEIT_START_DOCSTRING)
class FlaxBeitForMaskedImageModeling(FlaxBeitPreTrainedModel):
    module_class = FlaxBeitForMaskedImageModelingModule