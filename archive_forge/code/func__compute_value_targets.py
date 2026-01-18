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
def _compute_value_targets(self, *, hps: DreamerV3LearnerHyperparameters, rewards_t0_to_H_BxT: TensorType, intrinsic_rewards_t1_to_H_BxT: TensorType, continues_t0_to_H_BxT: TensorType, value_predictions_t0_to_H_BxT: TensorType) -> TensorType:
    """Helper method computing the value targets.

        All args are (H, BxT, ...) and in non-symlog'd (real) reward space.
        Non-symlog is important b/c log(a+b) != log(a) + log(b).
        See [1] eq. 8 and 10.
        Thus, targets are always returned in real (non-symlog'd space).
        They need to be re-symlog'd before computing the critic loss from them (b/c the
        critic produces predictions in symlog space).
        Note that the original B and T ranks together form the new batch dimension
        (folded into BxT) and the new time rank is the dream horizon (hence: [H, BxT]).

        Variable names nomenclature:
        `H`=1+horizon_H (start state + H steps dreamed),
        `BxT`=batch_size * batch_length (meaning the original trajectory time rank has
        been folded).

        Rewards, continues, and value predictions are all of shape [t0-H, BxT]
        (time-major), whereas returned targets are [t0 to H-1, B] (last timestep missing
        b/c the target value equals vf prediction in that location anyways.

        Args:
            hps: The DreamerV3LearnerHyperparameters to use.
            rewards_t0_to_H_BxT: The reward predictor's predictions over the
                dreamed trajectory t0 to H (and for the batch BxT).
            intrinsic_rewards_t1_to_H_BxT: The predicted intrinsic rewards over the
                dreamed trajectory t0 to H (and for the batch BxT).
            continues_t0_to_H_BxT: The continue predictor's predictions over the
                dreamed trajectory t0 to H (and for the batch BxT).
            value_predictions_t0_to_H_BxT: The critic's value predictions over the
                dreamed trajectory t0 to H (and for the batch BxT).

        Returns:
            The value targets in the shape: [t0toH-1, BxT]. Note that the last step (H)
            does not require a value target as it matches the critic's value prediction
            anyways.
        """
    rewards_t1_to_H_BxT = rewards_t0_to_H_BxT[1:]
    if intrinsic_rewards_t1_to_H_BxT is not None:
        rewards_t1_to_H_BxT += intrinsic_rewards_t1_to_H_BxT
    discount = continues_t0_to_H_BxT[1:] * hps.gamma
    Rs = [value_predictions_t0_to_H_BxT[-1]]
    intermediates = rewards_t1_to_H_BxT + discount * (1 - hps.gae_lambda) * value_predictions_t0_to_H_BxT[1:]
    for t in reversed(range(discount.shape[0])):
        Rs.append(intermediates[t] + discount[t] * hps.gae_lambda * Rs[-1])
    targets_t0toHm1_BxT = tf.stack(list(reversed(Rs))[:-1], axis=0)
    return targets_t0toHm1_BxT