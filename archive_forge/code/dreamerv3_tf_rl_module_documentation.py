from typing import Mapping, Any
from ray.rllib.algorithms.dreamerv3.dreamerv3_rl_module import DreamerV3RLModule
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
The tf-specific RLModule class for DreamerV3.

    Serves mainly as a thin-wrapper around the `DreamerModel` (a tf.keras.Model) class.
    