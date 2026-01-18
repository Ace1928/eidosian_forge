from typing import Dict, Optional, Tuple
import flax
import jax.numpy as jnp
from .utils import ModelOutput
@flax.struct.dataclass
class FlaxImageClassifierOutputWithNoAttention(ModelOutput):
    """
    Base class for outputs of image classification models.

    Args:
        logits (`jnp.ndarray` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings, if the model has an embedding layer, + one
            for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
            called feature maps) of the model at the output of each stage.
    """
    logits: jnp.ndarray = None
    hidden_states: Optional[Tuple[jnp.ndarray]] = None