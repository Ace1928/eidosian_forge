import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class PatchTSMixerLayer(nn.Module):
    """
    The `PatchTSMixer` layer that does all three kinds of mixing.

    Args:
        config (`PatchTSMixerConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: PatchTSMixerConfig):
        super().__init__()
        self.patch_mixer = PatchMixerBlock(config=config)
        self.feature_mixer = FeatureMixerBlock(config=config)
        self.mode = config.mode
        if config.mode == 'mix_channel':
            self.channel_feature_mixer = PatchTSMixerChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == 'mix_channel':
            hidden = self.channel_feature_mixer(hidden)
        hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)
        return hidden