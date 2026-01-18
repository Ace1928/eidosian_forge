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
def _update_module_target_networks(self, module_id: ModuleID, hps: AppoLearnerHyperparameters) -> None:
    module = self.module[module_id]
    target_current_network_pairs = module.get_target_network_pairs()
    for target_network, current_network in target_current_network_pairs:
        current_state_dict = current_network.state_dict()
        new_state_dict = {k: hps.tau * current_state_dict[k] + (1 - hps.tau) * v for k, v in target_network.state_dict().items()}
        target_network.load_state_dict(new_state_dict)