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
class TorchCNNEncoder(TorchModel, Encoder):

    def __init__(self, config: CNNEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        layers = []
        cnn = TorchCNN(input_dims=config.input_dims, cnn_filter_specifiers=config.cnn_filter_specifiers, cnn_activation=config.cnn_activation, cnn_use_layernorm=config.cnn_use_layernorm, cnn_use_bias=config.cnn_use_bias)
        layers.append(cnn)
        if config.flatten_at_end:
            layers.append(nn.Flatten())
        self.net = nn.Sequential(*layers)

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict({SampleBatch.OBS: TensorSpec('b, w, h, c', w=self.config.input_dims[0], h=self.config.input_dims[1], c=self.config.input_dims[2], framework='torch')})

    @override(Model)
    def get_output_specs(self) -> Optional[Spec]:
        return SpecDict({ENCODER_OUT: TensorSpec('b, d', d=self.config.output_dims[0], framework='torch') if self.config.flatten_at_end else TensorSpec('b, w, h, c', w=self.config.output_dims[0], h=self.config.output_dims[1], d=self.config.output_dims[2], framework='torch')})

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        return {ENCODER_OUT: self.net(inputs[SampleBatch.OBS])}