import copy
import inspect
import warnings
from functools import partial
from typing import Any, Dict, Optional, Union
import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from ..models.auto import (
from ..utils import ModelOutput, logging
from .configuration_utils import GenerationConfig
from .flax_logits_process import (
@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (`jnp.ndarray` of shape `(batch_size, max_length)`):
            The generated sequences.
    """
    sequences: jnp.ndarray = None