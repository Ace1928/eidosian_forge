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
def _validate_model_class(self):
    """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
    if not self.can_generate():
        generate_compatible_mappings = [FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING, FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING]
        generate_compatible_classes = set()
        for model_mapping in generate_compatible_mappings:
            supported_models = model_mapping.get(type(self.config), default=None)
            if supported_models is not None:
                generate_compatible_classes.add(supported_models.__name__)
        exception_message = f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as it doesn't have a language model head."
        if generate_compatible_classes:
            exception_message += f' Please use one of the following classes instead: {generate_compatible_classes}'
        raise TypeError(exception_message)