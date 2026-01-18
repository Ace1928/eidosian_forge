import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, SMALL_NUMBER
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
@DeveloperAPI
class MultiActionDistribution(TFActionDistribution):
    """Action distribution that operates on a set of actions.

    Args:
        inputs (Tensor list): A list of tensors from which to compute samples.
    """

    def __init__(self, inputs, model, *, child_distributions, input_lens, action_space, **kwargs):
        ActionDistribution.__init__(self, inputs, model)
        self.action_space_struct = get_base_struct_from_space(action_space)
        self.input_lens = np.array(input_lens, dtype=np.int32)
        split_inputs = tf.split(inputs, self.input_lens, axis=1)
        self.flat_child_distributions = tree.map_structure(lambda dist, input_: dist(input_, model, **kwargs), child_distributions, split_inputs)

    @override(ActionDistribution)
    def logp(self, x):
        if isinstance(x, (tf.Tensor, np.ndarray)):
            split_indices = []
            for dist in self.flat_child_distributions:
                if isinstance(dist, Categorical):
                    split_indices.append(1)
                elif isinstance(dist, MultiCategorical) and dist.action_space is not None:
                    split_indices.append(np.prod(dist.action_space.shape))
                else:
                    sample = dist.sample()
                    if len(sample.shape) == 1:
                        split_indices.append(1)
                    else:
                        split_indices.append(tf.shape(sample)[1])
            split_x = tf.split(x, split_indices, axis=1)
        else:
            split_x = tree.flatten(x)

        def map_(val, dist):
            if isinstance(dist, Categorical):
                val = tf.cast(tf.squeeze(val, axis=-1) if len(val.shape) > 1 else val, tf.int32)
            return dist.logp(val)
        flat_logps = tree.map_structure(map_, split_x, self.flat_child_distributions)
        return functools.reduce(lambda a, b: a + b, flat_logps)

    @override(ActionDistribution)
    def kl(self, other):
        kl_list = [d.kl(o) for d, o in zip(self.flat_child_distributions, other.flat_child_distributions)]
        return functools.reduce(lambda a, b: a + b, kl_list)

    @override(ActionDistribution)
    def entropy(self):
        entropy_list = [d.entropy() for d in self.flat_child_distributions]
        return functools.reduce(lambda a, b: a + b, entropy_list)

    @override(ActionDistribution)
    def sample(self):
        child_distributions = tree.unflatten_as(self.action_space_struct, self.flat_child_distributions)
        return tree.map_structure(lambda s: s.sample(), child_distributions)

    @override(ActionDistribution)
    def deterministic_sample(self):
        child_distributions = tree.unflatten_as(self.action_space_struct, self.flat_child_distributions)
        return tree.map_structure(lambda s: s.deterministic_sample(), child_distributions)

    @override(TFActionDistribution)
    def sampled_action_logp(self):
        p = self.flat_child_distributions[0].sampled_action_logp()
        for c in self.flat_child_distributions[1:]:
            p += c.sampled_action_logp()
        return p

    @override(ActionDistribution)
    def required_model_output_shape(self, action_space, model_config):
        return np.sum(self.input_lens, dtype=np.int32)