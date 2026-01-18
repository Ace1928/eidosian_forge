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
def _get_logits_warper(self, generation_config: GenerationConfig) -> FlaxLogitsProcessorList:
    """
        This class returns a [`FlaxLogitsProcessorList`] list object that contains all relevant [`FlaxLogitsWarper`]
        instances used for multinomial sampling.
        """
    warpers = FlaxLogitsProcessorList()
    if generation_config.temperature is not None and generation_config.temperature != 1.0:
        warpers.append(FlaxTemperatureLogitsWarper(generation_config.temperature))
    if generation_config.top_k is not None and generation_config.top_k != 0:
        warpers.append(FlaxTopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
    if generation_config.top_p is not None and generation_config.top_p < 1.0:
        warpers.append(FlaxTopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=1))
    return warpers