from typing import Any, Dict, Mapping
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.appo.appo_learner import (
from ray.rllib.algorithms.impala.tf.vtrace_tf_v2 import make_time_major, vtrace_tf2
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import TensorType
Implements APPO loss / update logic on top of ImpalaTfLearner.