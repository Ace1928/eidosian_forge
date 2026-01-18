from typing import Callable, Optional, Tuple
import flax
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
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_electra import ElectraConfig
class FlaxElectraSequenceSummary(nn.Module):
    """
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.summary = identity
        if hasattr(self.config, 'summary_use_proj') and self.config.summary_use_proj:
            if hasattr(self.config, 'summary_proj_to_labels') and self.config.summary_proj_to_labels and (self.config.num_labels > 0):
                num_classes = self.config.num_labels
            else:
                num_classes = self.config.hidden_size
            self.summary = nn.Dense(num_classes, dtype=self.dtype)
        activation_string = getattr(self.config, 'summary_activation', None)
        self.activation = ACT2FN[activation_string] if activation_string else lambda x: x
        self.first_dropout = identity
        if hasattr(self.config, 'summary_first_dropout') and self.config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(self.config.summary_first_dropout)
        self.last_dropout = identity
        if hasattr(self.config, 'summary_last_dropout') and self.config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(self.config.summary_last_dropout)

    def __call__(self, hidden_states, cls_index=None, deterministic: bool=True):
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`jnp.ndarray` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`jnp.ndarray` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `jnp.ndarray`: The summary of the sequence hidden states.
        """
        output = hidden_states[:, 0]
        output = self.first_dropout(output, deterministic=deterministic)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output, deterministic=deterministic)
        return output