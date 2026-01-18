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
def _make_time_major(policy, seq_lens, tensor):
    """Swaps batch and trajectory axis.

    Args:
        policy: Policy reference
        seq_lens: Sequence lengths if recurrent or None
        tensor: A tensor or list of tensors to reshape.
        trajectory item.

    Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
    """
    if isinstance(tensor, list):
        return [_make_time_major(policy, seq_lens, t) for t in tensor]
    if policy.is_recurrent():
        B = tf.shape(seq_lens)[0]
        T = tf.shape(tensor)[0] // B
    else:
        T = policy.config['rollout_fragment_length']
        B = tf.shape(tensor)[0] // T
    rs = tf.reshape(tensor, tf.concat([[B, T], tf.shape(tensor)[1:]], axis=0))
    res = tf.transpose(rs, [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))))
    return res