from typing import Any, Mapping
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
class DiscreteBCTorchModule(TorchRLModule):

    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        input_dim = self.config.observation_space.shape[0]
        hidden_dim = self.config.model_config_dict['fcnet_hiddens'][0]
        output_dim = self.config.action_space.n
        self.policy = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.input_dim = input_dim

    def get_train_action_dist_cls(self):
        return TorchCategorical

    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    def get_inference_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def output_specs_exploration(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_inference(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        return [SampleBatch.ACTION_DIST_INPUTS]

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        action_logits = self.policy(batch['obs'])
        return {SampleBatch.ACTION_DIST_INPUTS: action_logits}