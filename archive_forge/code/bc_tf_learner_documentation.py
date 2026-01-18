import logging
from typing import Mapping
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import TensorType
Implements tf-specific BC loss logic.

    This class implements the BC loss under `self.compute_loss_for_module()`.
    