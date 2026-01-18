import abc
from dataclasses import dataclass, field
import functools
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
@dataclass
class ActorCriticEncoderConfig(ModelConfig):
    """Configuration for an ActorCriticEncoder.

    The base encoder functions like other encoders in RLlib. It is wrapped by the
    ActorCriticEncoder to provides a shared encoder Model to use in RLModules that
    provides twofold outputs: one for the actor and one for the critic. See
    ModelConfig for usage details.

    Attributes:
        base_encoder_config: The configuration for the wrapped encoder(s).
        shared: Whether the base encoder is shared between the actor and critic.
    """
    base_encoder_config: ModelConfig = None
    shared: bool = True

    @_framework_implemented()
    def build(self, framework: str='torch') -> 'Encoder':
        if framework == 'torch':
            from ray.rllib.core.models.torch.encoder import TorchActorCriticEncoder, TorchStatefulActorCriticEncoder
            if isinstance(self.base_encoder_config, RecurrentEncoderConfig):
                return TorchStatefulActorCriticEncoder(self)
            else:
                return TorchActorCriticEncoder(self)
        else:
            from ray.rllib.core.models.tf.encoder import TfActorCriticEncoder, TfStatefulActorCriticEncoder
            if isinstance(self.base_encoder_config, RecurrentEncoderConfig):
                return TfStatefulActorCriticEncoder(self)
            else:
                return TfActorCriticEncoder(self)