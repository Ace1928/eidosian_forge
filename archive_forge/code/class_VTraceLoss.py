import numpy as np
import logging
import gymnasium as gym
from typing import Dict, List, Optional, Type, Union
from ray.rllib.algorithms.impala import vtrace_tf as vtrace
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import compute_bootstrap_value
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import explained_variance
from ray.rllib.policy.tf_mixins import GradStatsMixin, ValueNetworkMixin
from ray.rllib.utils.typing import (
class VTraceLoss:

    def __init__(self, actions, actions_logp, actions_entropy, dones, behaviour_action_logp, behaviour_logits, target_logits, discount, rewards, values, bootstrap_value, dist_class, model, valid_mask, config, vf_loss_coeff=0.5, entropy_coeff=0.01, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0):
        """Policy gradient loss with vtrace importance weighting.

        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.

        Args:
            actions: An int|float32 tensor of shape [T, B, ACTION_SPACE].
            actions_logp: A float32 tensor of shape [T, B].
            actions_entropy: A float32 tensor of shape [T, B].
            dones: A bool tensor of shape [T, B].
            behaviour_action_logp: Tensor of shape [T, B].
            behaviour_logits: A list with length of ACTION_SPACE of float32
                tensors of shapes
                [T, B, ACTION_SPACE[0]],
                ...,
                [T, B, ACTION_SPACE[-1]]
            target_logits: A list with length of ACTION_SPACE of float32
                tensors of shapes
                [T, B, ACTION_SPACE[0]],
                ...,
                [T, B, ACTION_SPACE[-1]]
            discount: A float32 scalar.
            rewards: A float32 tensor of shape [T, B].
            values: A float32 tensor of shape [T, B].
            bootstrap_value: A float32 tensor of shape [B].
            dist_class: action distribution class for logits.
            valid_mask: A bool tensor of valid RNN input elements (#2992).
            config: Algorithm config dict.
        """
        with tf.device('/cpu:0'):
            self.vtrace_returns = vtrace.multi_from_logits(behaviour_action_log_probs=behaviour_action_logp, behaviour_policy_logits=behaviour_logits, target_policy_logits=target_logits, actions=tf.unstack(actions, axis=2), discounts=tf.cast(~tf.cast(dones, tf.bool), tf.float32) * discount, rewards=rewards, values=values, bootstrap_value=bootstrap_value, dist_class=dist_class, model=model, clip_rho_threshold=tf.cast(clip_rho_threshold, tf.float32), clip_pg_rho_threshold=tf.cast(clip_pg_rho_threshold, tf.float32))
            self.value_targets = self.vtrace_returns.vs
        masked_pi_loss = tf.boolean_mask(actions_logp * self.vtrace_returns.pg_advantages, valid_mask)
        self.pi_loss = -tf.reduce_sum(masked_pi_loss)
        self.mean_pi_loss = -tf.reduce_mean(masked_pi_loss)
        delta = tf.boolean_mask(values - self.vtrace_returns.vs, valid_mask)
        delta_squarred = tf.math.square(delta)
        self.vf_loss = 0.5 * tf.reduce_sum(delta_squarred)
        self.mean_vf_loss = 0.5 * tf.reduce_mean(delta_squarred)
        masked_entropy = tf.boolean_mask(actions_entropy, valid_mask)
        self.entropy = tf.reduce_sum(masked_entropy)
        self.mean_entropy = tf.reduce_mean(masked_entropy)
        self.total_loss = self.pi_loss - self.entropy * entropy_coeff
        self.loss_wo_vf = self.total_loss
        if not config['_separate_vf_optimizer']:
            self.total_loss += self.vf_loss * vf_loss_coeff