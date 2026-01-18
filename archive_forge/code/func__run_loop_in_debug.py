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
@staticmethod
def _run_loop_in_debug(cond_fn, body_fn, init_state):
    """
        Run generation in untraced mode. This should only be used for debugging purposes.
        """
    state = init_state
    while cond_fn(state):
        state = body_fn(state)
    return state