from typing import List
import gymnasium as gym
from ray.rllib.models.tf.layers import NoisyLayer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def _layer(x):
    support_logits_per_action = tf.reshape(tensor=x, shape=(-1, self.action_space.n, num_atoms))
    support_prob_per_action = tf.nn.softmax(logits=support_logits_per_action)
    x = tf.reduce_sum(input_tensor=z * support_prob_per_action, axis=-1)
    logits = support_logits_per_action
    dist = support_prob_per_action
    return [x, z, support_logits_per_action, logits, dist]