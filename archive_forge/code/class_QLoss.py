from typing import Dict
import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.simple_q.utils import Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import get_categorical_class_with_temperature
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import LearningRateSchedule, TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration import ParameterNoise
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.tf_utils import (
from ray.rllib.utils.typing import AlgorithmConfigDict, ModelGradients, TensorType
class QLoss:

    def __init__(self, q_t_selected: TensorType, q_logits_t_selected: TensorType, q_tp1_best: TensorType, q_dist_tp1_best: TensorType, importance_weights: TensorType, rewards: TensorType, done_mask: TensorType, gamma: float=0.99, n_step: int=1, num_atoms: int=1, v_min: float=-10.0, v_max: float=10.0, loss_fn=huber_loss):
        if num_atoms > 1:
            z = tf.range(num_atoms, dtype=tf.float32)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)
            r_tau = tf.expand_dims(rewards, -1) + gamma ** n_step * tf.expand_dims(1.0 - done_mask, -1) * tf.expand_dims(z, 0)
            r_tau = tf.clip_by_value(r_tau, v_min, v_max)
            b = (r_tau - v_min) / ((v_max - v_min) / float(num_atoms - 1))
            lb = tf.floor(b)
            ub = tf.math.ceil(b)
            floor_equal_ceil = tf.cast(tf.less(ub - lb, 0.5), tf.float32)
            l_project = tf.one_hot(tf.cast(lb, dtype=tf.int32), num_atoms)
            u_project = tf.one_hot(tf.cast(ub, dtype=tf.int32), num_atoms)
            ml_delta = q_dist_tp1_best * (ub - b + floor_equal_ceil)
            mu_delta = q_dist_tp1_best * (b - lb)
            ml_delta = tf.reduce_sum(l_project * tf.expand_dims(ml_delta, -1), axis=1)
            mu_delta = tf.reduce_sum(u_project * tf.expand_dims(mu_delta, -1), axis=1)
            m = ml_delta + mu_delta
            self.td_error = tf.nn.softmax_cross_entropy_with_logits(labels=m, logits=q_logits_t_selected)
            self.loss = tf.reduce_mean(self.td_error * tf.cast(importance_weights, tf.float32))
            self.stats = {'mean_td_error': tf.reduce_mean(self.td_error)}
        else:
            q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best
            q_t_selected_target = rewards + gamma ** n_step * q_tp1_best_masked
            self.td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            self.loss = tf.reduce_mean(tf.cast(importance_weights, tf.float32) * loss_fn(self.td_error))
            self.stats = {'mean_q': tf.reduce_mean(q_t_selected), 'min_q': tf.reduce_min(q_t_selected), 'max_q': tf.reduce_max(q_t_selected), 'mean_td_error': tf.reduce_mean(self.td_error)}