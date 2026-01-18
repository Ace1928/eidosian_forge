from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig
@add_start_docstrings('\n    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for\n    ImageNet.\n    ', RESNET_START_DOCSTRING)
class FlaxResNetForImageClassification(FlaxResNetPreTrainedModel):
    module_class = FlaxResNetForImageClassificationModule