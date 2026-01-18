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
def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
    """Validates model kwargs for generation. Generate argument typos will also be caught here."""
    unused_model_args = []
    model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
    if 'kwargs' in model_args or 'model_kwargs' in model_args:
        model_args |= set(inspect.signature(self.__call__).parameters)
    for key, value in model_kwargs.items():
        if value is not None and key not in model_args:
            unused_model_args.append(key)
    if unused_model_args:
        raise ValueError(f'The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the generate arguments will also show up in this list)')