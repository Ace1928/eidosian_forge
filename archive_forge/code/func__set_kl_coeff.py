import logging
from typing import Dict, List
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.eager_tf_policy import EagerTFPolicy
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.typing import (
def _set_kl_coeff(self, new_kl_coeff):
    self.kl_coeff_val = new_kl_coeff
    if self.framework == 'tf':
        self.get_session().run(self._kl_coeff_update, feed_dict={self._kl_coeff_placeholder: self.kl_coeff_val})
    else:
        self.kl_coeff.assign(self.kl_coeff_val, read_value=False)