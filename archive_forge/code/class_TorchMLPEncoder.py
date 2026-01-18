from typing import Optional
import tree
from ray.rllib.core.models.base import (
from ray.rllib.core.models.base import Model, tokenize
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.core.models.specs.specs_base import TensorSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.primitives import TorchMLP, TorchCNN
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
class TorchMLPEncoder(TorchModel, Encoder):

    def __init__(self, config: MLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        self.net = TorchMLP(input_dim=config.input_dims[0], hidden_layer_dims=config.hidden_layer_dims, hidden_layer_activation=config.hidden_layer_activation, hidden_layer_use_layernorm=config.hidden_layer_use_layernorm, hidden_layer_use_bias=config.hidden_layer_use_bias, output_dim=config.output_layer_dim, output_activation=config.output_layer_activation, output_use_bias=config.output_layer_use_bias)

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict({SampleBatch.OBS: TensorSpec('b, d', d=self.config.input_dims[0], framework='torch')})

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict({ENCODER_OUT: TensorSpec('b, d', d=self.config.output_dims[0], framework='torch')})

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS])}