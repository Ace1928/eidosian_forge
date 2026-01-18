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
class VTraceClipGradients:
    """VTrace version of gradient computation logic."""

    def __init__(self):
        """No special initialization required."""
        pass

    def compute_gradients_fn(self, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
        if self.config.get('_enable_new_api_stack', False):
            trainable_variables = self.model.trainable_variables
        else:
            trainable_variables = self.model.trainable_variables()
        if self.config['_tf_policy_handles_more_than_one_loss']:
            optimizers = force_list(optimizer)
            losses = force_list(loss)
            assert len(optimizers) == len(losses)
            clipped_grads_and_vars = []
            for optim, loss_ in zip(optimizers, losses):
                grads_and_vars = optim.compute_gradients(loss_, trainable_variables)
                clipped_g_and_v = []
                for g, v in grads_and_vars:
                    if g is not None:
                        clipped_g, _ = tf.clip_by_global_norm([g], self.config['grad_clip'])
                        clipped_g_and_v.append((clipped_g[0], v))
                clipped_grads_and_vars.append(clipped_g_and_v)
            self.grads = [g for g_and_v in clipped_grads_and_vars for g, v in g_and_v]
        else:
            grads_and_vars = optimizer.compute_gradients(loss, self.model.trainable_variables())
            grads = [g for g, v in grads_and_vars]
            self.grads, _ = tf.clip_by_global_norm(grads, self.config['grad_clip'])
            clipped_grads_and_vars = list(zip(self.grads, trainable_variables))
        return clipped_grads_and_vars