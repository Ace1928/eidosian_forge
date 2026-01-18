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
def _merge_criteria_processor_list(self, default_list: FlaxLogitsProcessorList, custom_list: FlaxLogitsProcessorList) -> FlaxLogitsProcessorList:
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = 'logits processor'
                raise ValueError(f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to `generate`, but it has already been created with the values {default}. {default} has been created by passing the corresponding arguments to generate or by the model's config default values. If you just want to change the default values of {object_type} consider passing them as arguments to `generate` instead of using a custom {object_type}.")
    default_list.extend(custom_list)
    return default_list