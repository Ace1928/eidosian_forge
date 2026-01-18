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
def _create_fc_net(self, layer_dims, activation, name=None):
    """Given a list of layer dimensions (incl. input-dim), creates FC-net.

        Args:
            layer_dims (Tuple[int]): Tuple of layer dims, including the input
                dimension.
            activation: An activation specifier string (e.g. "relu").

        Examples:
            If layer_dims is [4,8,6] we'll have a two layer net: 4->8 (8 nodes)
            and 8->6 (6 nodes), where the second layer (6 nodes) does not have
            an activation anymore. 4 is the input dimension.
        """
    layers = [tf.keras.layers.Input(shape=(layer_dims[0],), name='{}_in'.format(name))] if self.framework != 'torch' else []
    for i in range(len(layer_dims) - 1):
        act = activation if i < len(layer_dims) - 2 else None
        if self.framework == 'torch':
            layers.append(SlimFC(in_size=layer_dims[i], out_size=layer_dims[i + 1], initializer=torch.nn.init.xavier_uniform_, activation_fn=act))
        else:
            layers.append(tf.keras.layers.Dense(units=layer_dims[i + 1], activation=get_activation_fn(act), name='{}_{}'.format(name, i)))
    if self.framework == 'torch':
        return nn.Sequential(*layers)
    else:
        return tf.keras.Sequential(layers)