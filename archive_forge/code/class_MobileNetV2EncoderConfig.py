from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.framework import try_import_torch
class MobileNetV2EncoderConfig(ModelConfig):
    output_dims = (1000,)
    freeze = True

    def build(self, framework):
        assert framework == 'torch', 'Unsupported framework `{}`!'.format(framework)
        return MobileNetV2Encoder(self)