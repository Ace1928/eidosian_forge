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
def _forward_shared(self, batch: NestedDict) -> Mapping[str, Any]:
    action_logits = self.policy(batch['obs'])
    return {SampleBatch.ACTION_DIST_INPUTS: action_logits}