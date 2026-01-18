from typing import Optional
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.reward_predictor_layer import (
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
Updates the EMA-copy of the critic according to the update formula:

        ema_net=(`ema_decay`*ema_net) + (1.0-`ema_decay`)*critic_net
        