from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.representation_layer import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
Predict the RSSM's z^(t+1), given h(t), z^(t), and a(t).

    Disagreement (stddev) between the N networks in this model on what the next z^ would
    be are used to produce intrinsic rewards for enhanced, curiosity-based exploration.

    TODO
    