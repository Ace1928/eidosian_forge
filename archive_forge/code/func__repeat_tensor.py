from functools import partial
import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.typing import (
def _repeat_tensor(t: TensorType, n: int):
    t_rep = tf.expand_dims(t, 1)
    multiples = tf.concat([[1, n], tf.tile([1], tf.expand_dims(tf.rank(t) - 1, 0))], 0)
    t_rep = tf.tile(t_rep, multiples)
    t_rep = tf.reshape(t_rep, tf.concat([[-1], tf.shape(t)[1:]], 0))
    return t_rep