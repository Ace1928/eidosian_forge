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
class MARWILLoss:

    def __init__(self, policy: Policy, value_estimates: TensorType, action_dist: ActionDistribution, train_batch: SampleBatch, vf_loss_coeff: float, beta: float):
        logprobs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
        if beta != 0.0:
            cumulative_rewards = train_batch[Postprocessing.ADVANTAGES]
            adv = cumulative_rewards - value_estimates
            adv_squared = tf.reduce_mean(tf.math.square(adv))
            self.v_loss = 0.5 * adv_squared
            rate = policy.config['moving_average_sqd_adv_norm_update_rate']
            if policy.config['framework'] == 'tf2':
                update_term = adv_squared - policy._moving_average_sqd_adv_norm
                policy._moving_average_sqd_adv_norm.assign_add(rate * update_term)
                c = tf.math.sqrt(policy._moving_average_sqd_adv_norm)
                exp_advs = tf.math.exp(beta * (adv / (1e-08 + c)))
            else:
                update_adv_norm = tf1.assign_add(ref=policy._moving_average_sqd_adv_norm, value=rate * (adv_squared - policy._moving_average_sqd_adv_norm))
                with tf1.control_dependencies([update_adv_norm]):
                    exp_advs = tf.math.exp(beta * tf.math.divide(adv, 1e-08 + tf.math.sqrt(policy._moving_average_sqd_adv_norm)))
            exp_advs = tf.stop_gradient(exp_advs)
            self.explained_variance = tf.reduce_mean(explained_variance(cumulative_rewards, value_estimates))
        else:
            self.v_loss = tf.constant(0.0)
            exp_advs = 1.0
        logstd_coeff = policy.config['bc_logstd_coeff']
        if logstd_coeff > 0.0:
            logstds = tf.reduce_sum(action_dist.log_std, axis=1)
        else:
            logstds = 0.0
        self.p_loss = -1.0 * tf.reduce_mean(exp_advs * (logprobs + logstd_coeff * logstds))
        self.total_loss = self.p_loss + vf_loss_coeff * self.v_loss