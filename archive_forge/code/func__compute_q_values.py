import logging
from typing import Dict, List, Tuple, Type, Union
from ray.rllib.algorithms.simple_q.utils import make_q_models
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy_v2 import DynamicTFPolicyV2
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_utils import huber_loss
from ray.rllib.utils.typing import LocalOptimizer, ModelGradients, TensorType
def _compute_q_values(self, model: ModelV2, obs_batch: TensorType, is_training=None) -> TensorType:
    _is_training = is_training if is_training is not None else self._get_is_training_placeholder()
    model_out, _ = model(SampleBatch(obs=obs_batch, _is_training=_is_training), [], None)
    return model_out