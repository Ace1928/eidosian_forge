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
@add_start_docstrings('\n    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', REGNET_START_DOCSTRING)
class FlaxRegNetForImageClassification(FlaxRegNetPreTrainedModel):
    module_class = FlaxRegNetForImageClassificationModule