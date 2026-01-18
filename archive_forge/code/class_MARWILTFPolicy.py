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
class MARWILTFPolicy(ValueNetworkMixin, PostprocessAdvantages, base):

    def __init__(self, observation_space, action_space, config, existing_model=None, existing_inputs=None):
        base.enable_eager_execution_if_necessary()
        base.__init__(self, observation_space, action_space, config, existing_inputs=existing_inputs, existing_model=existing_model)
        ValueNetworkMixin.__init__(self, config)
        PostprocessAdvantages.__init__(self)
        if config['beta'] != 0.0:
            self._moving_average_sqd_adv_norm = get_variable(config['moving_average_sqd_adv_norm_start'], framework='tf', tf_name='moving_average_of_advantage_norm', trainable=False)
        self.maybe_initialize_optimizer_and_loss()

    @override(base)
    def loss(self, model: Union[ModelV2, 'tf.keras.Model'], dist_class: Type[TFActionDistribution], train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        model_out, _ = model(train_batch)
        action_dist = dist_class(model_out, model)
        value_estimates = model.value_function()
        self._marwil_loss = MARWILLoss(self, value_estimates, action_dist, train_batch, self.config['vf_coeff'], self.config['beta'])
        return self._marwil_loss.total_loss

    @override(base)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = {'policy_loss': self._marwil_loss.p_loss, 'total_loss': self._marwil_loss.total_loss}
        if self.config['beta'] != 0.0:
            stats['moving_average_sqd_adv_norm'] = self._moving_average_sqd_adv_norm
            stats['vf_explained_var'] = self._marwil_loss.explained_variance
            stats['vf_loss'] = self._marwil_loss.v_loss
        return stats

    @override(base)
    def compute_gradients_fn(self, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
        return compute_gradients(self, optimizer, loss)