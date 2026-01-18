import logging
from typing import Any, Dict, List, Optional, Type, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, get_variable
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.utils.typing import (
class PostprocessAdvantages:
    """Marwil's custom trajectory post-processing mixin."""

    def __init__(self):
        pass

    def postprocess_trajectory(self, sample_batch: SampleBatch, other_agent_batches: Optional[Dict[Any, SampleBatch]]=None, episode: Optional['Episode']=None):
        sample_batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
        if sample_batch[SampleBatch.TERMINATEDS][-1]:
            last_r = 0.0
        else:
            index = 'last' if SampleBatch.NEXT_OBS in sample_batch else -1
            input_dict = sample_batch.get_single_step_input_dict(self.view_requirements, index=index)
            last_r = self._value(**input_dict)
        return compute_advantages(sample_batch, last_r, self.config['gamma'], use_gae=False, use_critic=False)