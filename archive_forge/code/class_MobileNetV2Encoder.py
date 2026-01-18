from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.framework import try_import_torch
class MobileNetV2Encoder(TorchModel, Encoder):
    """A MobileNet v2 encoder for RLlib."""

    def __init__(self, config):
        super().__init__(config)
        self.net = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        if config.freeze:
            for p in self.net.parameters():
                p.requires_grad = False

    def _forward(self, input_dict, **kwargs):
        return {ENCODER_OUT: self.net(input_dict['obs'])}