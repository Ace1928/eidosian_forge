from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_big_bird import BigBirdConfig
@add_start_docstrings('\n    BigBird Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a\n    softmax) e.g. for RocStories/SWAG tasks.\n    ', BIG_BIRD_START_DOCSTRING)
class FlaxBigBirdForMultipleChoice(FlaxBigBirdPreTrainedModel):
    module_class = FlaxBigBirdForMultipleChoiceModule

    def __init__(self, config: BigBirdConfig, input_shape: Optional[tuple]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if config.attention_type == 'block_sparse' and input_shape is None:
            input_shape = (1, 1, 12 * config.block_size)
        elif input_shape is None:
            input_shape = (1, 1)
        super().__init__(config, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)