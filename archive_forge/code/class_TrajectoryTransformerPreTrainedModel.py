import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig
class TrajectoryTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TrajectoryTransformerConfig
    load_tf_weights = load_tf_weights_in_trajectory_transformer
    base_model_prefix = 'trajectory_transformer'
    main_input_name = 'trajectories'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, EinLinear):
            for i in range(module.n_models):
                nn.init.kaiming_uniform_(module.weight[i], a=math.sqrt(5) / self.config.kaiming_initializer_range)
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight[i])
                    bound = 1 / math.sqrt(fan_in) * self.config.initializer_range
                    nn.init.uniform_(module.bias[i], -bound, bound)