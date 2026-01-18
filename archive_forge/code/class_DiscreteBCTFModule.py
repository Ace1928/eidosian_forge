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
class DiscreteBCTFModule(TfRLModule):

    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        input_dim = self.config.observation_space.shape[0]
        hidden_dim = self.config.model_config_dict['fcnet_hiddens'][0]
        output_dim = self.config.action_space.n
        layers = []
        layers.append(tf.keras.Input(shape=(input_dim,)))
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.Dense(hidden_dim))
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.Dense(output_dim))
        self.policy = tf.keras.Sequential(layers)
        self._input_dim = input_dim

    def get_train_action_dist_cls(self):
        return TfCategorical

    def get_exploration_action_dist_cls(self):
        return TfCategorical

    def get_inference_action_dist_cls(self):
        return TfCategorical

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    def _forward_shared(self, batch: NestedDict) -> Mapping[str, Any]:
        action_logits = self.policy(batch['obs'])
        return {SampleBatch.ACTION_DIST_INPUTS: action_logits}

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_shared(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_shared(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_shared(batch)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        return {'policy': self.policy.get_weights()}

    @override(RLModule)
    def set_state(self, state: Mapping[str, Any]) -> None:
        self.policy.set_weights(state['policy'])