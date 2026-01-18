import logging
import numpy as np
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        hiddens = list(model_config.get('fcnet_hiddens', [])) + list(model_config.get('post_fcnet_hiddens', []))
        activation = model_config.get('fcnet_activation')
        if not model_config.get('fcnet_hiddens', []):
            activation = model_config.get('post_fcnet_activation')
        no_final_linear = model_config.get('no_final_linear')
        self.vf_share_layers = model_config.get('vf_share_layers')
        self.free_log_std = model_config.get('free_log_std')
        if self.free_log_std:
            assert num_outputs % 2 == 0, ('num_outputs must be divisible by two', num_outputs)
            num_outputs = num_outputs // 2
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        for size in hiddens[:-1]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=size, initializer=normc_initializer(1.0), activation_fn=activation))
            prev_layer_size = size
        if no_final_linear and num_outputs:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=num_outputs, initializer=normc_initializer(1.0), activation_fn=activation))
            prev_layer_size = num_outputs
        else:
            if len(hiddens) > 0:
                layers.append(SlimFC(in_size=prev_layer_size, out_size=hiddens[-1], initializer=normc_initializer(1.0), activation_fn=activation))
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(in_size=prev_layer_size, out_size=num_outputs, initializer=normc_initializer(0.01), activation_fn=None)
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)
        self._hidden_layers = nn.Sequential(*layers)
        self._value_branch_separate = None
        if not self.vf_share_layers:
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(SlimFC(in_size=prev_vf_layer_size, out_size=size, activation_fn=activation, initializer=normc_initializer(1.0)))
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)
        self._value_branch = SlimFC(in_size=prev_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)
        self._features = None
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict['obs_flat'].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return (logits, state)

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, 'must call forward() first'
        if self._value_branch_separate:
            out = self._value_branch(self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out