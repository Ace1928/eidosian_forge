import functools
from typing import Optional
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.checker import SpecCheckingError
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.tf.base import TfModel
from ray.rllib.core.models.tf.primitives import TfCNNTranspose, TfMLP
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.annotations import override
class TfMLPHead(TfModel):

    def __init__(self, config: MLPHeadConfig) -> None:
        TfModel.__init__(self, config)
        self.net = TfMLP(input_dim=config.input_dims[0], hidden_layer_dims=config.hidden_layer_dims, hidden_layer_activation=config.hidden_layer_activation, hidden_layer_use_layernorm=config.hidden_layer_use_layernorm, hidden_layer_use_bias=config.hidden_layer_use_bias, output_dim=config.output_layer_dim, output_activation=config.output_layer_activation, output_use_bias=config.output_layer_use_bias)

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return TensorSpec('b, d', d=self.config.input_dims[0], framework='tf2')

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return TensorSpec('b, d', d=self.config.output_dims[0], framework='tf2')

    @override(Model)
    @auto_fold_unfold_time('input_specs')
    def _forward(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        return self.net(inputs)