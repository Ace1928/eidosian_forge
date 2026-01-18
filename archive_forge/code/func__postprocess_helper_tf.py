from gymnasium.spaces import Discrete, MultiDiscrete, Space
import numpy as np
from typing import Optional, Tuple, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, MultiCategorical
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import NullContextManager
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder, one_hot as tf_one_hot
from ray.rllib.utils.torch_utils import one_hot
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
def _postprocess_helper_tf(self, obs, next_obs, actions):
    with tf.GradientTape() if self.framework != 'tf' else NullContextManager() as tape:
        phis, _ = self.model._curiosity_feature_net({SampleBatch.OBS: tf.concat([obs, next_obs], axis=0)})
        phi, next_phi = tf.split(phis, 2)
        predicted_next_phi = self.model._curiosity_forward_fcnet(tf.concat([phi, tf_one_hot(actions, self.action_space)], axis=-1))
        forward_l2_norm_sqared = 0.5 * tf.reduce_sum(tf.square(predicted_next_phi - next_phi), axis=-1)
        forward_loss = tf.reduce_mean(forward_l2_norm_sqared)
        phi_cat_next_phi = tf.concat([phi, next_phi], axis=-1)
        dist_inputs = self.model._curiosity_inverse_fcnet(phi_cat_next_phi)
        action_dist = Categorical(dist_inputs, self.model) if isinstance(self.action_space, Discrete) else MultiCategorical(dist_inputs, self.model, self.action_space.nvec)
        inverse_loss = -action_dist.logp(tf.convert_to_tensor(actions))
        inverse_loss = tf.reduce_mean(inverse_loss)
        loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
    if self.framework != 'tf':
        grads = tape.gradient(loss, self._optimizer_var_list)
        grads_and_vars = [(g, v) for g, v in zip(grads, self._optimizer_var_list) if g is not None]
        update_op = self._optimizer.apply_gradients(grads_and_vars)
    else:
        update_op = self._optimizer.minimize(loss, var_list=self._optimizer_var_list)
    return (forward_l2_norm_sqared, update_op)