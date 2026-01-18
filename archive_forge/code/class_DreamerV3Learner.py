from dataclasses import dataclass
from typing import Any, DefaultDict, Dict
from ray.rllib.core.learner.learner import Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
class DreamerV3Learner(Learner):
    """DreamerV3 specific Learner class.

    Only implements the `additional_update_for_module()` method to define the logic
    for updating the critic EMA-copy after each training step.
    """

    @override(Learner)
    def compile_results(self, *, batch: MultiAgentBatch, fwd_out: Dict[str, Any], loss_per_module: Dict[str, TensorType], metrics_per_module: DefaultDict[ModuleID, Dict[str, Any]]) -> Dict[str, Any]:
        results = super().compile_results(batch=batch, fwd_out=fwd_out, loss_per_module=loss_per_module, metrics_per_module=metrics_per_module)
        if self.hps.report_images_and_videos:
            for module_id, res in results.items():
                if module_id in fwd_out:
                    res['WORLD_MODEL_fwd_out_obs_distribution_means_BxT'] = fwd_out[module_id]['obs_distribution_means_BxT']
        return results

    @override(Learner)
    def additional_update_for_module(self, *, module_id: ModuleID, hps: DreamerV3LearnerHyperparameters, timestep: int) -> Dict[str, Any]:
        """Updates the EMA weights of the critic network."""
        results = super().additional_update_for_module(module_id=module_id, hps=hps, timestep=timestep)
        self.module[module_id].critic.update_ema()
        return results