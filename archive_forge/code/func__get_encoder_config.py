import enum
import functools
from typing import Optional
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import (
@classmethod
def _get_encoder_config(cls, observation_space: gym.Space, model_config_dict: dict, action_space: gym.Space=None, view_requirements=None) -> ModelConfig:
    """Returns an EncoderConfig for the given input_space and model_config_dict.

        Encoders are usually used in RLModules to transform the input space into a
        latent space that is then fed to the heads. The returned EncoderConfig
        objects correspond to the built-in Encoder classes in RLlib.
        For example, for a simple 1D-Box input_space, RLlib offers an
        MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
        this method to produce specific EncoderConfigs for your custom Models.

        The following input spaces lead to the following configs:
        - 1D-Box: MLPEncoderConfig
        - 3D-Box: CNNEncoderConfig
        # TODO (Artur): Support more spaces here
        # ...

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            action_space: The action space to use if actions are to be encoded. This
                is commonly the case for LSTM models.
            view_requirements: The view requirements to use if anything else than
                observation_space or action_space is to be encoded. This signifies an
                advanced use case.

        Returns:
            The encoder config.
        """
    model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}
    activation = model_config_dict['fcnet_activation']
    output_activation = model_config_dict['fcnet_activation']
    fcnet_hiddens = model_config_dict['fcnet_hiddens']
    encoder_latent_dim = model_config_dict['encoder_latent_dim'] or fcnet_hiddens[-1]
    use_lstm = model_config_dict['use_lstm']
    use_attention = model_config_dict['use_attention']
    if use_lstm:
        encoder_config = RecurrentEncoderConfig(input_dims=observation_space.shape, recurrent_layer_type='lstm', hidden_dim=model_config_dict['lstm_cell_size'], batch_major=not model_config_dict['_time_major'], num_layers=1, tokenizer_config=cls.get_tokenizer_config(observation_space, model_config_dict, view_requirements))
    elif use_attention:
        raise NotImplementedError
    elif isinstance(observation_space, Box) and len(observation_space.shape) == 1:
        if model_config_dict['encoder_latent_dim']:
            hidden_layer_dims = model_config_dict['fcnet_hiddens']
        else:
            hidden_layer_dims = model_config_dict['fcnet_hiddens'][:-1]
        encoder_config = MLPEncoderConfig(input_dims=observation_space.shape, hidden_layer_dims=hidden_layer_dims, hidden_layer_activation=activation, output_layer_dim=encoder_latent_dim, output_layer_activation=output_activation)
    elif isinstance(observation_space, Box) and len(observation_space.shape) == 3:
        if not model_config_dict.get('conv_filters'):
            model_config_dict['conv_filters'] = get_filter_config(observation_space.shape)
        encoder_config = CNNEncoderConfig(input_dims=observation_space.shape, cnn_filter_specifiers=model_config_dict['conv_filters'], cnn_activation=model_config_dict['conv_activation'], cnn_use_layernorm=model_config_dict.get('conv_use_layernorm', False))
    elif isinstance(observation_space, Box) and len(observation_space.shape) == 2:
        raise ValueError(f'No default encoder config for obs space={observation_space}, lstm={use_lstm} and attention={use_attention} found. 2D Box spaces are not supported. They should be either flattened to a 1D Box space or enhanced to be a 3D box space.')
    else:
        raise ValueError(f'No default encoder config for obs space={observation_space}, lstm={use_lstm} and attention={use_attention} found.')
    return encoder_config