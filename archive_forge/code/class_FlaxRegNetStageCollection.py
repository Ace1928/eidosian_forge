from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import RegNetConfig
from transformers.modeling_flax_outputs import (
from transformers.modeling_flax_utils import (
from transformers.utils import (
class FlaxRegNetStageCollection(nn.Module):
    config: RegNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        in_out_channels = zip(self.config.hidden_sizes, self.config.hidden_sizes[1:])
        stages = [FlaxRegNetStage(self.config, self.config.embedding_size, self.config.hidden_sizes[0], stride=2 if self.config.downsample_in_first_stage else 1, depth=self.config.depths[0], dtype=self.dtype, name='0')]
        for i, ((in_channels, out_channels), depth) in enumerate(zip(in_out_channels, self.config.depths[1:])):
            stages.append(FlaxRegNetStage(self.config, in_channels, out_channels, depth=depth, dtype=self.dtype, name=str(i + 1)))
        self.stages = stages

    def __call__(self, hidden_state: jnp.ndarray, output_hidden_states: bool=False, deterministic: bool=True) -> FlaxBaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state.transpose(0, 3, 1, 2),)
            hidden_state = stage_module(hidden_state, deterministic=deterministic)
        return (hidden_state, hidden_states)