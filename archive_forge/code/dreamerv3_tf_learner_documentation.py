from typing import Any, Dict, Mapping, Tuple
import gymnasium as gym
from ray.rllib.algorithms.dreamerv3.dreamerv3_learner import (
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.core.learner.learner import ParamDict
from ray.rllib.core.learner.tf.tf_learner import TfLearner
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.tf_utils import symlog, two_hot, clip_gradients
from ray.rllib.utils.typing import TensorType
Helper method computing the scaled value targets.

        Args:
            module_id: The module_id to compute value targets for.
            hps: The DreamerV3LearnerHyperparameters to use.
            value_targets_t0_to_Hm1_BxT: The value targets computed by
                `self._compute_value_targets` in the shape of (t0 to H-1, BxT)
                and of type float32.
            value_predictions_t0_to_Hm1_BxT: The critic's value predictions over the
                dreamed trajectories (w/o the last timestep). The shape of this
                tensor is (t0 to H-1, BxT) and the type is float32.

        Returns:
            The scaled value targets used by the actor for REINFORCE policy updates
            (using scaled advantages). See [1] eq. 12 for more details.
        