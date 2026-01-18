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
def _adapt_logits_for_beam_search(self, logits):
    """
        This function can be overwritten in the specific modeling_flax_<model-name>.py classes to allow for custom beam
        search behavior. Note that the only model that overwrites this method is [`~transformes.FlaxMarianMTModel`].
        """
    return logits