import tensorflow as tf
from typing import Any, Mapping
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.tf.tf_distributions import TfCategorical
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
class BCTfMultiAgentModuleWithSharedEncoder(MultiAgentRLModule):

    def __init__(self, config: MultiAgentRLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        module_specs = self.config.modules
        module_spec = next(iter(module_specs.values()))
        global_dim = module_spec.observation_space['global'].shape[0]
        hidden_dim = module_spec.model_config_dict['fcnet_hiddens'][0]
        shared_encoder = tf.keras.Sequential([tf.keras.Input(shape=(global_dim,)), tf.keras.layers.ReLU(), tf.keras.layers.Dense(hidden_dim)])
        for module_id, module_spec in module_specs.items():
            self._rl_modules[module_id] = module_spec.module_class(encoder=shared_encoder, local_dim=module_spec.observation_space['local'].shape[0], hidden_dim=hidden_dim, action_dim=module_spec.action_space.n)

    def serialize(self):
        raise NotImplementedError

    def deserialize(self, data):
        raise NotImplementedError