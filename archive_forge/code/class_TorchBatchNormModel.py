import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.misc import (
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
class TorchBatchNormModel(TorchModelV2, nn.Module):
    """Example of a TorchModelV2 using batch normalization."""
    capture_index = 0

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        for size in [256, 256]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=size, initializer=torch_normc_initializer(1.0), activation_fn=nn.ReLU))
            prev_layer_size = size
            layers.append(nn.BatchNorm1d(prev_layer_size))
        self._logits = SlimFC(in_size=prev_layer_size, out_size=self.num_outputs, initializer=torch_normc_initializer(0.01), activation_fn=None)
        self._value_branch = SlimFC(in_size=prev_layer_size, out_size=1, initializer=torch_normc_initializer(1.0), activation_fn=None)
        self._hidden_layers = nn.Sequential(*layers)
        self._hidden_out = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, SampleBatch):
            is_training = bool(input_dict.is_training)
        else:
            is_training = bool(input_dict.get('is_training', False))
        self._hidden_layers.train(mode=is_training)
        self._hidden_out = self._hidden_layers(input_dict['obs'])
        logits = self._logits(self._hidden_out)
        return (logits, [])

    @override(ModelV2)
    def value_function(self):
        assert self._hidden_out is not None, 'must call forward first!'
        return torch.reshape(self._value_branch(self._hidden_out), [-1])