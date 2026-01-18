from typing import Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import (
def _get_tf_exploration_action(self, action_dist, explore):
    action = tf.argmax(tf.cond(explore, lambda: action_dist.inputs, lambda: self.model.value_function()), axis=-1)
    return (action, None)