from dataclasses import dataclass
import torch
import torch.nn as nn
from xformers.components import Activation, build_activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig
from . import register_feedforward
@dataclass
class MlpConfig(FeedforwardConfig):
    hidden_layer_multiplier: int
    bias: bool