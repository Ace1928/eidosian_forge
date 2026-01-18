from typing import Any, Dict, Mapping
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.appo.appo_learner import (
from ray.rllib.algorithms.impala.torch.vtrace_torch_v2 import (
from ray.rllib.core.learner.learner import POLICY_LOSS_KEY, VF_LOSS_KEY, ENTROPY_KEY
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.rl_module.marl_module import ModuleID, MultiAgentRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import (
from ray.rllib.core.rl_module.rl_module_with_target_networks_interface import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import TensorType
@override(AppoLearner)
def _update_module_kl_coeff(self, module_id: ModuleID, hps: AppoLearnerHyperparameters, sampled_kl: float) -> Dict[str, Any]:
    kl_coeff_var = self.curr_kl_coeffs_per_module[module_id]
    if sampled_kl > 2.0 * hps.kl_target:
        kl_coeff_var.data *= 1.5
    elif sampled_kl < 0.5 * hps.kl_target:
        kl_coeff_var.data *= 0.5
    return {LEARNER_RESULTS_CURR_KL_COEFF_KEY: kl_coeff_var.item()}